"""
Kirk's Approximation

Spread option pricing:
- Options on difference of two assets
- Crack spread options
- Calendar spread options
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from finceptquantlib.core.foundations.types import OptionType
from finceptquantlib.pricing.analytical.black_scholes import norm_cdf


# =============================================================================
# Kirk's Approximation
# =============================================================================

def kirk_spread_call(
    F1: float,
    F2: float,
    K: float,
    r: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    T: float
) -> float:
    """
    Kirk's approximation for spread call option.

    Call on spread: max(F1 - F2 - K, 0)

    Approximates the spread option as a vanilla option on a modified underlying.

    Args:
        F1: Forward price of asset 1
        F2: Forward price of asset 2
        K: Strike (spread strike)
        r: Risk-free rate
        sigma1: Volatility of asset 1
        sigma2: Volatility of asset 2
        rho: Correlation between assets
        T: Time to maturity

    Returns:
        Spread call price

    Note:
        Works best when K ≈ 0 (at-the-money spread options).
        Less accurate for deep in/out-of-the-money.
    """
    if T <= 0:
        return max(F1 - F2 - K, 0)

    # Kirk's adjustment
    # Treat as call on F1 with strike (F2 + K), adjusted volatility
    adj_strike = F2 + K

    if adj_strike <= 0:
        # Spread is always positive enough
        return math.exp(-r * T) * (F1 - F2 - K)

    # Adjusted volatility
    sigma_ratio = sigma2 * F2 / adj_strike

    sigma_kirk_sq = sigma1 ** 2 - 2 * rho * sigma1 * sigma_ratio + sigma_ratio ** 2
    sigma_kirk = math.sqrt(max(0.0, sigma_kirk_sq))

    if sigma_kirk <= 0 or F1 <= 0:
        return math.exp(-r * T) * max(F1 - adj_strike, 0)

    # Black-Scholes call on ratio
    d1 = (math.log(F1 / adj_strike) + 0.5 * sigma_kirk ** 2 * T) / (sigma_kirk * math.sqrt(T))
    d2 = d1 - sigma_kirk * math.sqrt(T)

    return math.exp(-r * T) * (F1 * norm_cdf(d1) - adj_strike * norm_cdf(d2))


def kirk_spread_put(
    F1: float,
    F2: float,
    K: float,
    r: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    T: float
) -> float:
    """
    Kirk's approximation for spread put option.

    Put on spread: max(K - (F1 - F2), 0) = max(K + F2 - F1, 0)
    """
    if T <= 0:
        return max(K - (F1 - F2), 0)

    adj_strike = F2 + K

    if adj_strike <= 0:
        return 0.0

    sigma_ratio = sigma2 * F2 / adj_strike

    sigma_kirk_sq = sigma1 ** 2 - 2 * rho * sigma1 * sigma_ratio + sigma_ratio ** 2
    sigma_kirk = math.sqrt(max(0.0, sigma_kirk_sq))

    if sigma_kirk <= 0 or F1 <= 0:
        return math.exp(-r * T) * max(adj_strike - F1, 0)

    d1 = (math.log(F1 / adj_strike) + 0.5 * sigma_kirk ** 2 * T) / (sigma_kirk * math.sqrt(T))
    d2 = d1 - sigma_kirk * math.sqrt(T)

    return math.exp(-r * T) * (adj_strike * norm_cdf(-d2) - F1 * norm_cdf(-d1))


def kirk_spread_price(
    F1: float,
    F2: float,
    K: float,
    r: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    T: float,
    option_type: OptionType
) -> float:
    """Kirk spread option price."""
    if option_type == OptionType.CALL:
        return kirk_spread_call(F1, F2, K, r, sigma1, sigma2, rho, T)
    else:
        return kirk_spread_put(F1, F2, K, r, sigma1, sigma2, rho, T)


# =============================================================================
# Greeks
# =============================================================================

def kirk_delta1(
    F1: float, F2: float, K: float, r: float,
    sigma1: float, sigma2: float, rho: float, T: float,
    option_type: OptionType
) -> float:
    """
    Delta with respect to F1.

    Approximated via finite difference.
    """
    eps = max(abs(F1) * 1e-4, 1e-8)
    price_up = kirk_spread_price(F1 + eps, F2, K, r, sigma1, sigma2, rho, T, option_type)
    price_down = kirk_spread_price(F1 - eps, F2, K, r, sigma1, sigma2, rho, T, option_type)
    return (price_up - price_down) / (2 * eps)


def kirk_delta2(
    F1: float, F2: float, K: float, r: float,
    sigma1: float, sigma2: float, rho: float, T: float,
    option_type: OptionType
) -> float:
    """Delta with respect to F2."""
    eps = max(abs(F2) * 1e-4, 1e-8)
    price_up = kirk_spread_price(F1, F2 + eps, K, r, sigma1, sigma2, rho, T, option_type)
    price_down = kirk_spread_price(F1, F2 - eps, K, r, sigma1, sigma2, rho, T, option_type)
    return (price_up - price_down) / (2 * eps)


@dataclass
class KirkGreeks:
    """Kirk spread option Greeks."""
    delta1: float
    delta2: float
    gamma1: float
    gamma2: float
    cross_gamma: float
    vega1: float
    vega2: float


def kirk_greeks(
    F1: float, F2: float, K: float, r: float,
    sigma1: float, sigma2: float, rho: float, T: float,
    option_type: OptionType
) -> KirkGreeks:
    """Compute Kirk Greeks via finite difference."""
    eps_f = max(min(abs(F1), abs(F2)) * 1e-4, 1e-8)
    eps_s = 0.001  # 0.1% vol bump

    # Deltas
    delta1 = kirk_delta1(F1, F2, K, r, sigma1, sigma2, rho, T, option_type)
    delta2 = kirk_delta2(F1, F2, K, r, sigma1, sigma2, rho, T, option_type)

    # Gammas
    price = kirk_spread_price(F1, F2, K, r, sigma1, sigma2, rho, T, option_type)
    price_up1 = kirk_spread_price(F1 + eps_f, F2, K, r, sigma1, sigma2, rho, T, option_type)
    price_down1 = kirk_spread_price(F1 - eps_f, F2, K, r, sigma1, sigma2, rho, T, option_type)
    gamma1 = (price_up1 - 2 * price + price_down1) / (eps_f ** 2)

    price_up2 = kirk_spread_price(F1, F2 + eps_f, K, r, sigma1, sigma2, rho, T, option_type)
    price_down2 = kirk_spread_price(F1, F2 - eps_f, K, r, sigma1, sigma2, rho, T, option_type)
    gamma2 = (price_up2 - 2 * price + price_down2) / (eps_f ** 2)

    # Cross gamma
    price_upup = kirk_spread_price(F1 + eps_f, F2 + eps_f, K, r, sigma1, sigma2, rho, T, option_type)
    price_updown = kirk_spread_price(F1 + eps_f, F2 - eps_f, K, r, sigma1, sigma2, rho, T, option_type)
    price_downup = kirk_spread_price(F1 - eps_f, F2 + eps_f, K, r, sigma1, sigma2, rho, T, option_type)
    price_downdown = kirk_spread_price(F1 - eps_f, F2 - eps_f, K, r, sigma1, sigma2, rho, T, option_type)
    cross_gamma = (price_upup - price_updown - price_downup + price_downdown) / (4 * eps_f ** 2)

    # Vegas
    price_vol1_up = kirk_spread_price(F1, F2, K, r, sigma1 + eps_s, sigma2, rho, T, option_type)
    vega1 = (price_vol1_up - price) / eps_s / 100

    price_vol2_up = kirk_spread_price(F1, F2, K, r, sigma1, sigma2 + eps_s, rho, T, option_type)
    vega2 = (price_vol2_up - price) / eps_s / 100

    return KirkGreeks(
        delta1=delta1,
        delta2=delta2,
        gamma1=gamma1,
        gamma2=gamma2,
        cross_gamma=cross_gamma,
        vega1=vega1,
        vega2=vega2
    )


# =============================================================================
# Margrabe's Formula (Exchange Options)
# =============================================================================

def margrabe_call(
    S1: float,
    S2: float,
    q1: float,
    q2: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    T: float
) -> float:
    """
    Margrabe's formula for exchange option.

    Option to exchange asset 2 for asset 1:
    max(S1 - S2, 0)

    This is a spread option with K = 0.

    Args:
        S1: Price of asset 1
        S2: Price of asset 2
        q1: Dividend yield of asset 1
        q2: Dividend yield of asset 2
        sigma1, sigma2: Volatilities
        rho: Correlation
        T: Time to maturity

    Returns:
        Exchange option price
    """
    if T <= 0:
        return max(S1 - S2, 0)

    sigma = math.sqrt(max(0.0, sigma1 ** 2 - 2 * rho * sigma1 * sigma2 + sigma2 ** 2))

    if sigma <= 0:
        return max(S1 * math.exp(-q1 * T) - S2 * math.exp(-q2 * T), 0)

    d1 = (math.log(S1 / S2) + (q2 - q1 + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    return S1 * math.exp(-q1 * T) * norm_cdf(d1) - S2 * math.exp(-q2 * T) * norm_cdf(d2)


def margrabe_put(
    S1: float,
    S2: float,
    q1: float,
    q2: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    T: float
) -> float:
    """
    Margrabe put (option to exchange S1 for S2).

    max(S2 - S1, 0)
    """
    # By symmetry
    return margrabe_call(S2, S1, q2, q1, sigma2, sigma1, rho, T)


# =============================================================================
# Basket Option Approximations
# =============================================================================

def basket_call_levy(
    weights: list,
    forwards: list,
    sigmas: list,
    correlations: list,  # Flat list, row-major
    K: float,
    r: float,
    T: float
) -> float:
    """
    Levy (moment matching) approximation for basket call.

    Matches first two moments of basket to lognormal.

    Args:
        weights: Asset weights (sum to 1)
        forwards: Forward prices
        sigmas: Volatilities
        correlations: Correlation matrix (flattened)
        K: Strike
        r: Risk-free rate
        T: Time to maturity

    Returns:
        Basket call price
    """
    n = len(weights)

    # First moment: E[B] = Σ w_i * F_i
    M1 = sum(weights[i] * forwards[i] for i in range(n))

    # Second moment for lognormal matching
    M2 = 0.0
    for i in range(n):
        for j in range(n):
            rho_ij = correlations[i * n + j] if i != j else 1.0
            M2 += (weights[i] * weights[j] * forwards[i] * forwards[j] *
                   math.exp(rho_ij * sigmas[i] * sigmas[j] * T))

    # Match to lognormal
    # M1 = F_B, M2 = F_B^2 * exp(sigma_B^2 * T)
    if M2 <= M1 ** 2:
        sigma_B = 0.0
    else:
        sigma_B = math.sqrt(math.log(M2 / M1 ** 2) / T)

    F_B = M1

    # Black-76 call
    if sigma_B <= 0 or T <= 0:
        return math.exp(-r * T) * max(F_B - K, 0)

    d1 = (math.log(F_B / K) + 0.5 * sigma_B ** 2 * T) / (sigma_B * math.sqrt(T))
    d2 = d1 - sigma_B * math.sqrt(T)

    return math.exp(-r * T) * (F_B * norm_cdf(d1) - K * norm_cdf(d2))
