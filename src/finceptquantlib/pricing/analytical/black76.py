"""
Black-76 Model

Pricing for options on futures and forwards:
- European options on futures
- Caps and floors
- Swaptions (single payment)

Parameter Conventions:
---------------------
This module uses short parameter names for backward compatibility:
- F: forward/futures price
- K: strike price
- r: risk-free rate (decimal)
- sigma: volatility (decimal)
- T: time to maturity (years)

For standardized naming, use the wrapper functions with full names.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from finceptquantlib.core.foundations.types import OptionType
from finceptquantlib.core.utilities.distributions import norm_cdf, norm_pdf
from finceptquantlib.pricing.analytical._implied_vol import solve_implied_vol


# =============================================================================
# Black-76 Formulas
# =============================================================================

def black76_d1(F: float, K: float, sigma: float, T: float) -> float:
    """
    d1 for Black-76 model.

    d1 = (ln(F/K) + σ²T/2) / (σ√T)
    """
    if T <= 0 or sigma <= 0:
        return float('inf') if F > K else float('-inf')
    if F <= 0 or K <= 0:
        raise ValueError(f"F and K must be positive for lognormal model (F={F}, K={K})")
    return (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))


def black76_d2(F: float, K: float, sigma: float, T: float) -> float:
    """d2 = d1 - σ√T."""
    if T <= 0 or sigma <= 0:
        return float('inf') if F > K else float('-inf')
    return black76_d1(F, K, sigma, T) - sigma * math.sqrt(T)


def black76_call(F: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Black-76 call option price.

    C = e^(-rT) * (F*N(d1) - K*N(d2))

    Args:
        F: Forward/futures price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        T: Time to maturity

    Returns:
        Call option price
    """
    if T <= 0:
        return max(F - K, 0)
    if sigma <= 0:
        return math.exp(-r * T) * max(F - K, 0)

    d1 = black76_d1(F, K, sigma, T)
    d2 = d1 - sigma * math.sqrt(T)

    return math.exp(-r * T) * (F * norm_cdf(d1) - K * norm_cdf(d2))


def black76_put(F: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Black-76 put option price.

    P = e^(-rT) * (K*N(-d2) - F*N(-d1))
    """
    if T <= 0:
        return max(K - F, 0)
    if sigma <= 0:
        return math.exp(-r * T) * max(K - F, 0)

    d1 = black76_d1(F, K, sigma, T)
    d2 = d1 - sigma * math.sqrt(T)

    return math.exp(-r * T) * (K * norm_cdf(-d2) - F * norm_cdf(-d1))


def black76_price(
    F: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: OptionType
) -> float:
    """Black-76 option price."""
    if option_type == OptionType.CALL:
        return black76_call(F, K, r, sigma, T)
    else:
        return black76_put(F, K, r, sigma, T)


# =============================================================================
# Greeks
# =============================================================================

def black76_delta(
    F: float, K: float, r: float, sigma: float, T: float,
    option_type: OptionType
) -> float:
    """
    Black-76 delta: ∂V/∂F

    Call: e^(-rT) * N(d1)
    Put: e^(-rT) * (N(d1) - 1)
    """
    if T <= 0:
        if option_type == OptionType.CALL:
            return math.exp(-r * T) if F > K else 0.0
        else:
            return -math.exp(-r * T) if F < K else 0.0

    d1 = black76_d1(F, K, sigma, T)
    discount = math.exp(-r * T)

    if option_type == OptionType.CALL:
        return discount * norm_cdf(d1)
    else:
        return discount * (norm_cdf(d1) - 1)


def black76_gamma(F: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Black-76 gamma: ∂²V/∂F²

    Γ = e^(-rT) * N'(d1) / (F * σ * √T)
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = black76_d1(F, K, sigma, T)
    return math.exp(-r * T) * norm_pdf(d1) / (F * sigma * math.sqrt(T))


def black76_vega(F: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Black-76 vega: ∂V/∂σ (per 1% vol change)

    ν = e^(-rT) * F * N'(d1) * √T / 100
    """
    if T <= 0:
        return 0.0

    d1 = black76_d1(F, K, sigma, T)
    return math.exp(-r * T) * F * norm_pdf(d1) * math.sqrt(T) / 100


def black76_theta(
    F: float, K: float, r: float, sigma: float, T: float,
    option_type: OptionType
) -> float:
    """Black-76 theta (daily)."""
    if T <= 0:
        return 0.0

    d1 = black76_d1(F, K, sigma, T)
    d2 = d1 - sigma * math.sqrt(T)
    sqrt_T = math.sqrt(T)
    discount = math.exp(-r * T)

    first_term = -discount * F * norm_pdf(d1) * sigma / (2 * sqrt_T)

    # V = e^{-rT} * Black(F, T): as time passes the discounting unwinds, adding r * V
    if option_type == OptionType.CALL:
        theta_annual = first_term + r * discount * (F * norm_cdf(d1) - K * norm_cdf(d2))
    else:
        theta_annual = first_term + r * discount * (K * norm_cdf(-d2) - F * norm_cdf(-d1))

    return theta_annual / 365


@dataclass
class Black76Greeks:
    """Black-76 Greeks."""
    delta: float
    gamma: float
    vega: float
    theta: float


def black76_greeks(
    F: float, K: float, r: float, sigma: float, T: float,
    option_type: OptionType
) -> Black76Greeks:
    """Compute all Black-76 Greeks."""
    return Black76Greeks(
        delta=black76_delta(F, K, r, sigma, T, option_type),
        gamma=black76_gamma(F, K, r, sigma, T),
        vega=black76_vega(F, K, r, sigma, T),
        theta=black76_theta(F, K, r, sigma, T, option_type)
    )


# =============================================================================
# Implied Volatility
# =============================================================================

def black76_implied_vol(
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
    Black-76 implied volatility using Newton-Raphson.
    """
    if T <= 0:
        raise ValueError("Cannot compute implied vol with T <= 0")
    if F <= 0 or K <= 0:
        raise ValueError("Black76 implied vol requires positive forward and strike")

    # Initial guess using Brenner-Subrahmanyam approximation on the time value
    intrinsic = max(F - K, 0) if option_type == OptionType.CALL else max(K - F, 0)
    time_value = price - math.exp(-r * T) * intrinsic
    guess = math.sqrt(2 * math.pi / T) * time_value / (F * math.exp(-r * T)) if time_value > 0 else 0.2

    return solve_implied_vol(
        price_fn=lambda sigma: black76_price(F, K, r, sigma, T, option_type),
        vega_fn=lambda sigma: black76_vega(F, K, r, sigma, T) * 100,  # undo per-1% scaling
        target=price,
        guess=guess,
        lo=1e-6,
        hi=5.0,
        tol=tol,
        max_iter=max_iter,
        model="Black76",
    )


# =============================================================================
# Caps and Floors
# =============================================================================

def caplet_price(
    forward_rate: float,
    strike: float,
    discount_factor: float,
    sigma: float,
    T_start: float,
    T_end: float,
    notional: float = 1.0
) -> float:
    """
    Caplet price using Black-76.

    Caplet = τ * N * DF * Black76Call(F, K, σ, T)

    Args:
        forward_rate: Forward rate for the period
        strike: Cap strike rate
        discount_factor: DF to payment date
        sigma: Forward rate volatility
        T_start: Option expiry (start of rate period)
        T_end: Payment date (end of rate period)
        notional: Notional amount

    Returns:
        Caplet price
    """
    tau = T_end - T_start  # Day count fraction (simplified)

    if T_start <= 0:
        return tau * notional * discount_factor * max(forward_rate - strike, 0)
    if sigma <= 0:
        return tau * notional * discount_factor * max(forward_rate - strike, 0)

    # Black-76 with r=0 since we already have DF
    d1 = black76_d1(forward_rate, strike, sigma, T_start)
    d2 = d1 - sigma * math.sqrt(T_start)

    black_value = forward_rate * norm_cdf(d1) - strike * norm_cdf(d2)

    return tau * notional * discount_factor * black_value


def floorlet_price(
    forward_rate: float,
    strike: float,
    discount_factor: float,
    sigma: float,
    T_start: float,
    T_end: float,
    notional: float = 1.0
) -> float:
    """
    Floorlet price using Black-76.

    Floorlet = τ * N * DF * Black76Put(F, K, σ, T)
    """
    tau = T_end - T_start

    if T_start <= 0:
        return tau * notional * discount_factor * max(strike - forward_rate, 0)
    if sigma <= 0:
        return tau * notional * discount_factor * max(strike - forward_rate, 0)

    d1 = black76_d1(forward_rate, strike, sigma, T_start)
    d2 = d1 - sigma * math.sqrt(T_start)

    black_value = strike * norm_cdf(-d2) - forward_rate * norm_cdf(-d1)

    return tau * notional * discount_factor * black_value


# =============================================================================
# Swaption (Single Payment Approximation)
# =============================================================================

def swaption_price(
    forward_swap_rate: float,
    strike: float,
    annuity: float,
    sigma: float,
    T_expiry: float,
    is_payer: bool = True,
    notional: float = 1.0
) -> float:
    """
    European swaption price using Black-76.

    Payer swaption = A * Black76Call(S, K, σ, T)
    Receiver swaption = A * Black76Put(S, K, σ, T)

    where:
    - S = forward swap rate
    - A = swap annuity (PV01)

    Args:
        forward_swap_rate: Forward swap rate
        strike: Swaption strike
        annuity: Swap annuity factor
        sigma: Swap rate volatility
        T_expiry: Swaption expiry
        is_payer: True for payer, False for receiver
        notional: Notional amount
    """
    if T_expiry <= 0:
        if is_payer:
            return notional * annuity * max(forward_swap_rate - strike, 0)
        else:
            return notional * annuity * max(strike - forward_swap_rate, 0)

    d1 = black76_d1(forward_swap_rate, strike, sigma, T_expiry)
    d2 = d1 - sigma * math.sqrt(T_expiry)

    if is_payer:
        black_value = forward_swap_rate * norm_cdf(d1) - strike * norm_cdf(d2)
    else:
        black_value = strike * norm_cdf(-d2) - forward_swap_rate * norm_cdf(-d1)

    return notional * annuity * black_value


# =============================================================================
# Standardized API
# =============================================================================

@dataclass
class Black76Result:
    """Comprehensive Black-76 pricing result with standardized field names."""
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float

    # Input parameters
    forward: float = 0.0
    strike: float = 0.0
    risk_free_rate: float = 0.0
    volatility: float = 0.0
    time_to_maturity: float = 0.0
    option_type: Optional[OptionType] = None


def black76_option_price(
    forward: float,
    strike: float,
    risk_free_rate: float,
    volatility: float,
    time_to_maturity: float,
    option_type: OptionType,
) -> float:
    """
    Black-76 option price with standardized parameters.

    Args:
        forward: Forward/futures price
        strike: Strike price
        risk_free_rate: Risk-free rate (decimal)
        volatility: Lognormal volatility (decimal)
        time_to_maturity: Time to maturity in years
        option_type: CALL or PUT

    Returns:
        Option price
    """
    return black76_price(forward, strike, risk_free_rate, volatility, time_to_maturity, option_type)


def black76_option_greeks(
    forward: float,
    strike: float,
    risk_free_rate: float,
    volatility: float,
    time_to_maturity: float,
    option_type: OptionType,
) -> Black76Result:
    """
    Compute Black-76 price and Greeks with standardized parameters.

    Args:
        forward: Forward/futures price
        strike: Strike price
        risk_free_rate: Risk-free rate (decimal)
        volatility: Lognormal volatility (decimal)
        time_to_maturity: Time to maturity in years
        option_type: CALL or PUT

    Returns:
        Black76Result with price and all Greeks
    """
    price = black76_price(forward, strike, risk_free_rate, volatility, time_to_maturity, option_type)
    greeks = black76_greeks(forward, strike, risk_free_rate, volatility, time_to_maturity, option_type)

    return Black76Result(
        price=price,
        delta=greeks.delta,
        gamma=greeks.gamma,
        vega=greeks.vega,
        theta=greeks.theta,
        forward=forward,
        strike=strike,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_maturity=time_to_maturity,
        option_type=option_type,
    )


def black76_option_implied_volatility(
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
    Black-76 implied volatility with standardized parameters.

    Args:
        market_price: Observed market price
        forward: Forward/futures price
        strike: Strike price
        risk_free_rate: Risk-free rate (decimal)
        time_to_maturity: Time to maturity in years
        option_type: CALL or PUT
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations

    Returns:
        Implied volatility (decimal)
    """
    return black76_implied_vol(
        market_price, forward, strike, risk_free_rate,
        time_to_maturity, option_type, tolerance, max_iterations
    )
