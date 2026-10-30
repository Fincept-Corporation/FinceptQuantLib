"""
Black-Scholes Model

Complete Black-Scholes-Merton pricing formulas:
- European call and put prices
- All Greeks (delta, gamma, vega, theta, rho)
- Implied volatility
- Binary/digital options

Parameter Conventions:
---------------------
This module uses short parameter names for backward compatibility:
- S: spot price
- K: strike price
- r: risk-free rate (decimal, e.g., 0.05 for 5%)
- q: dividend yield (decimal)
- sigma: volatility (decimal, e.g., 0.20 for 20%)
- T: time to maturity (years)

For standardized naming, use the wrapper functions with full names:
- black_scholes_price(spot, strike, risk_free_rate, dividend_yield, volatility, time_to_maturity, option_type)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from finceptquantlib.core.foundations.types import OptionType
from finceptquantlib.core.utilities.distributions import norm_cdf, norm_pdf
from finceptquantlib.pricing.analytical._implied_vol import solve_implied_vol


# =============================================================================
# Black-Scholes Formulas
# =============================================================================

def d1(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """
    Compute d1 parameter.

    d1 = (ln(S/K) + (r - q + σ²/2)T) / (σ√T)
    """
    if T <= 0 or sigma <= 0:
        return float('inf') if S > K else float('-inf')
    if S <= 0 or K <= 0:
        raise ValueError(f"S and K must be positive for lognormal model (S={S}, K={K})")
    return (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def d2(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """
    Compute d2 parameter.

    d2 = d1 - σ√T
    """
    if T <= 0 or sigma <= 0:
        return float('inf') if S > K else float('-inf')
    if S <= 0 or K <= 0:
        raise ValueError(f"S and K must be positive for lognormal model (S={S}, K={K})")
    return d1(S, K, r, q, sigma, T) - sigma * math.sqrt(T)


def bs_call(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """
    Black-Scholes European call price.

    C = S*e^(-qT)*N(d1) - K*e^(-rT)*N(d2)

    Args:
        S: Spot price
        K: Strike price
        r: Risk-free rate (continuous)
        q: Dividend yield (continuous)
        sigma: Volatility
        T: Time to maturity

    Returns:
        Call option price
    """
    if T <= 0:
        return max(S - K, 0)
    if sigma <= 0:
        return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0)

    d_1 = d1(S, K, r, q, sigma, T)
    d_2 = d_1 - sigma * math.sqrt(T)

    return S * math.exp(-q * T) * norm_cdf(d_1) - K * math.exp(-r * T) * norm_cdf(d_2)


def bs_put(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """
    Black-Scholes European put price.

    P = K*e^(-rT)*N(-d2) - S*e^(-qT)*N(-d1)
    """
    if T <= 0:
        return max(K - S, 0)
    if sigma <= 0:
        return max(K * math.exp(-r * T) - S * math.exp(-q * T), 0)

    d_1 = d1(S, K, r, q, sigma, T)
    d_2 = d_1 - sigma * math.sqrt(T)

    return K * math.exp(-r * T) * norm_cdf(-d_2) - S * math.exp(-q * T) * norm_cdf(-d_1)


def bs_price(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    option_type: OptionType
) -> float:
    """
    Black-Scholes option price.

    Args:
        S: Spot price
        K: Strike price
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        T: Time to maturity
        option_type: CALL or PUT
    """
    if option_type == OptionType.CALL:
        return bs_call(S, K, r, q, sigma, T)
    else:
        return bs_put(S, K, r, q, sigma, T)


# =============================================================================
# Greeks
# =============================================================================

def bs_delta(
    S: float, K: float, r: float, q: float, sigma: float, T: float,
    option_type: OptionType
) -> float:
    """
    Option delta: ∂V/∂S

    Call: e^(-qT) * N(d1)
    Put: e^(-qT) * (N(d1) - 1)
    """
    if T <= 0:
        if option_type == OptionType.CALL:
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d_1 = d1(S, K, r, q, sigma, T)
    discount = math.exp(-q * T)

    if option_type == OptionType.CALL:
        return discount * norm_cdf(d_1)
    else:
        return discount * (norm_cdf(d_1) - 1)


def bs_gamma(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """
    Option gamma: ∂²V/∂S²

    Γ = e^(-qT) * N'(d1) / (S * σ * √T)

    Same for calls and puts.
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    d_1 = d1(S, K, r, q, sigma, T)
    return math.exp(-q * T) * norm_pdf(d_1) / (S * sigma * math.sqrt(T))


def bs_vega(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """
    Option vega: ∂V/∂σ

    ν = S * e^(-qT) * N'(d1) * √T

    Same for calls and puts. Returned per 1% vol change (divide by 100).
    """
    if T <= 0:
        return 0.0

    d_1 = d1(S, K, r, q, sigma, T)
    return S * math.exp(-q * T) * norm_pdf(d_1) * math.sqrt(T) / 100


def bs_theta(
    S: float, K: float, r: float, q: float, sigma: float, T: float,
    option_type: OptionType
) -> float:
    """
    Option theta: -∂V/∂T (time decay per day).

    Returns daily theta (divide annual by 365).
    """
    if T <= 0:
        return 0.0

    d_1 = d1(S, K, r, q, sigma, T)
    d_2 = d_1 - sigma * math.sqrt(T)
    sqrt_T = math.sqrt(T)

    first_term = -S * math.exp(-q * T) * norm_pdf(d_1) * sigma / (2 * sqrt_T)

    if option_type == OptionType.CALL:
        theta_annual = (first_term +
                       q * S * math.exp(-q * T) * norm_cdf(d_1) -
                       r * K * math.exp(-r * T) * norm_cdf(d_2))
    else:
        theta_annual = (first_term -
                       q * S * math.exp(-q * T) * norm_cdf(-d_1) +
                       r * K * math.exp(-r * T) * norm_cdf(-d_2))

    return theta_annual / 365


def bs_rho(
    S: float, K: float, r: float, q: float, sigma: float, T: float,
    option_type: OptionType
) -> float:
    """
    Option rho: ∂V/∂r (per 1% rate change).

    Call: K * T * e^(-rT) * N(d2) / 100
    Put: -K * T * e^(-rT) * N(-d2) / 100
    """
    if T <= 0:
        return 0.0

    d_2 = d2(S, K, r, q, sigma, T)

    if option_type == OptionType.CALL:
        return K * T * math.exp(-r * T) * norm_cdf(d_2) / 100
    else:
        return -K * T * math.exp(-r * T) * norm_cdf(-d_2) / 100


def bs_vanna(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """
    Vanna: ∂²V/∂S∂σ = ∂Δ/∂σ

    Vanna = -e^(-qT) * N'(d1) * d2 / σ
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    d_1 = d1(S, K, r, q, sigma, T)
    d_2 = d_1 - sigma * math.sqrt(T)

    return -math.exp(-q * T) * norm_pdf(d_1) * d_2 / sigma


def bs_volga(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """
    Volga (vomma): ∂²V/∂σ² = ∂ν/∂σ

    Volga = ν * d1 * d2 / σ
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    d_1 = d1(S, K, r, q, sigma, T)
    d_2 = d_1 - sigma * math.sqrt(T)
    vega = bs_vega(S, K, r, q, sigma, T) * 100  # Undo the /100

    return vega * d_1 * d_2 / sigma / 100


@dataclass
class BSGreeks:
    """All Black-Scholes Greeks."""
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    vanna: float = 0.0
    volga: float = 0.0


def bs_greeks(
    S: float, K: float, r: float, q: float, sigma: float, T: float,
    option_type: OptionType
) -> BSGreeks:
    """Compute all Greeks at once."""
    return BSGreeks(
        delta=bs_delta(S, K, r, q, sigma, T, option_type),
        gamma=bs_gamma(S, K, r, q, sigma, T),
        vega=bs_vega(S, K, r, q, sigma, T),
        theta=bs_theta(S, K, r, q, sigma, T, option_type),
        rho=bs_rho(S, K, r, q, sigma, T, option_type),
        vanna=bs_vanna(S, K, r, q, sigma, T),
        volga=bs_volga(S, K, r, q, sigma, T)
    )


# =============================================================================
# Implied Volatility
# =============================================================================

def bs_implied_vol(
    price: float,
    S: float,
    K: float,
    r: float,
    q: float,
    T: float,
    option_type: OptionType,
    tol: float = 1e-8,
    max_iter: int = 100
) -> float:
    """
    Compute implied volatility using Newton-Raphson.

    Args:
        price: Market option price
        S, K, r, q, T: Option parameters
        option_type: CALL or PUT
        tol: Convergence tolerance
        max_iter: Maximum iterations

    Returns:
        Implied volatility

    Raises:
        ValueError: If no solution found
    """
    if T <= 0:
        raise ValueError("Cannot compute implied vol with T <= 0")
    if S <= 0:
        raise ValueError("Cannot compute implied vol with S <= 0")

    # Initial guess using Brenner-Subrahmanyam approximation
    guess = math.sqrt(2 * math.pi / T) * price / S

    return solve_implied_vol(
        price_fn=lambda sigma: bs_price(S, K, r, q, sigma, T, option_type),
        vega_fn=lambda sigma: bs_vega(S, K, r, q, sigma, T) * 100,  # undo per-1% scaling
        target=price,
        guess=guess,
        lo=1e-6,
        hi=5.0,
        tol=tol,
        max_iter=max_iter,
        model="Black-Scholes",
    )


# =============================================================================
# Digital/Binary Options
# =============================================================================

def bs_digital_call(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """
    Cash-or-nothing digital call price.

    Pays 1 if S_T > K, else 0.
    Price = e^(-rT) * N(d2)
    """
    if T <= 0:
        return math.exp(-r * T) if S > K else 0.0

    d_2 = d2(S, K, r, q, sigma, T)
    return math.exp(-r * T) * norm_cdf(d_2)


def bs_digital_put(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """
    Cash-or-nothing digital put price.

    Pays 1 if S_T < K, else 0.
    Price = e^(-rT) * N(-d2)
    """
    if T <= 0:
        return math.exp(-r * T) if S < K else 0.0

    d_2 = d2(S, K, r, q, sigma, T)
    return math.exp(-r * T) * norm_cdf(-d_2)


def bs_asset_or_nothing_call(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """
    Asset-or-nothing digital call.

    Pays S_T if S_T > K, else 0.
    Price = S * e^(-qT) * N(d1)
    """
    if T <= 0:
        return S if S > K else 0.0

    d_1 = d1(S, K, r, q, sigma, T)
    return S * math.exp(-q * T) * norm_cdf(d_1)


def bs_asset_or_nothing_put(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """
    Asset-or-nothing digital put.

    Pays S_T if S_T < K, else 0.
    """
    if T <= 0:
        return S if S < K else 0.0

    d_1 = d1(S, K, r, q, sigma, T)
    return S * math.exp(-q * T) * norm_cdf(-d_1)


# =============================================================================
# Standardized API (with full parameter names)
# =============================================================================

@dataclass
class BlackScholesResult:
    """
    Comprehensive Black-Scholes pricing result.

    Uses standardized field names across all pricing models.
    """
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    vanna: float = 0.0
    volga: float = 0.0

    # Input parameters (for reference)
    spot: float = 0.0
    strike: float = 0.0
    risk_free_rate: float = 0.0
    dividend_yield: float = 0.0
    volatility: float = 0.0
    time_to_maturity: float = 0.0
    option_type: Optional[OptionType] = None


def black_scholes_price(
    spot: float,
    strike: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    time_to_maturity: float,
    option_type: OptionType,
    *,
    # Aliases for backward compatibility
    S: Optional[float] = None,
    K: Optional[float] = None,
    r: Optional[float] = None,
    q: Optional[float] = None,
    sigma: Optional[float] = None,
    T: Optional[float] = None,
) -> float:
    """
    Black-Scholes European option price with standardized parameters.

    Args:
        spot: Current spot price
        strike: Strike price
        risk_free_rate: Risk-free interest rate (decimal, e.g., 0.05 for 5%)
        dividend_yield: Continuous dividend yield (decimal)
        volatility: Volatility (decimal, e.g., 0.20 for 20%)
        time_to_maturity: Time to maturity in years
        option_type: CALL or PUT

        (Deprecated aliases: S, K, r, q, sigma, T)

    Returns:
        Option price

    Example:
        >>> black_scholes_price(
        ...     spot=100, strike=100, risk_free_rate=0.05,
        ...     dividend_yield=0.02, volatility=0.20,
        ...     time_to_maturity=1.0, option_type=OptionType.CALL
        ... )
        9.2270...
    """
    # Handle aliases for backward compatibility
    _S = S if S is not None else spot
    _K = K if K is not None else strike
    _r = r if r is not None else risk_free_rate
    _q = q if q is not None else dividend_yield
    _sigma = sigma if sigma is not None else volatility
    _T = T if T is not None else time_to_maturity

    return bs_price(_S, _K, _r, _q, _sigma, _T, option_type)


def black_scholes_greeks(
    spot: float,
    strike: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    time_to_maturity: float,
    option_type: OptionType,
) -> BlackScholesResult:
    """
    Compute Black-Scholes price and all Greeks with standardized parameters.

    Args:
        spot: Current spot price
        strike: Strike price
        risk_free_rate: Risk-free interest rate (decimal)
        dividend_yield: Continuous dividend yield (decimal)
        volatility: Volatility (decimal)
        time_to_maturity: Time to maturity in years
        option_type: CALL or PUT

    Returns:
        BlackScholesResult with price and all Greeks
    """
    price = bs_price(spot, strike, risk_free_rate, dividend_yield, volatility, time_to_maturity, option_type)
    greeks = bs_greeks(spot, strike, risk_free_rate, dividend_yield, volatility, time_to_maturity, option_type)

    return BlackScholesResult(
        price=price,
        delta=greeks.delta,
        gamma=greeks.gamma,
        vega=greeks.vega,
        theta=greeks.theta,
        rho=greeks.rho,
        vanna=greeks.vanna,
        volga=greeks.volga,
        spot=spot,
        strike=strike,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        time_to_maturity=time_to_maturity,
        option_type=option_type,
    )


def black_scholes_implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    risk_free_rate: float,
    dividend_yield: float,
    time_to_maturity: float,
    option_type: OptionType,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> float:
    """
    Compute implied volatility with standardized parameters.

    Args:
        market_price: Observed market price
        spot: Current spot price
        strike: Strike price
        risk_free_rate: Risk-free interest rate (decimal)
        dividend_yield: Continuous dividend yield (decimal)
        time_to_maturity: Time to maturity in years
        option_type: CALL or PUT
        tolerance: Convergence tolerance
        max_iterations: Maximum Newton-Raphson iterations

    Returns:
        Implied volatility (decimal)

    Raises:
        ValueError: If implied vol calculation doesn't converge
    """
    return bs_implied_vol(
        market_price, spot, strike, risk_free_rate, dividend_yield,
        time_to_maturity, option_type, tolerance, max_iterations
    )
