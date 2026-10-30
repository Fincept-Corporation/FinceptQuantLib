"""
Sensitivity Calculations (Greeks)

Provides functions for calculating:
- First-order sensitivities (Delta)
- Second-order sensitivities (Gamma, Vanna, Volga)
- Time and rate sensitivities (Theta, Rho)
- Key Rate Durations
- Bucket Deltas
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, List, Dict, Callable, Tuple

from finceptquantlib.curves.curve import Curve


@dataclass
class GreeksResult:
    """Full Greeks for an instrument."""
    delta: float          # ∂V/∂S
    gamma: float          # ∂²V/∂S²
    vega: float           # ∂V/∂σ
    theta: float          # ∂V/∂t
    rho: float            # ∂V/∂r
    vanna: Optional[float] = None  # ∂²V/∂S∂σ
    volga: Optional[float] = None  # ∂²V/∂σ²
    charm: Optional[float] = None  # ∂²V/∂t∂S


def calculate_delta(
    pricing_func: Callable[[float], float],
    spot: float,
    bump_size: float = 0.01,
    method: str = "central",
) -> float:
    """
    Calculate delta using finite difference.

    Args:
        pricing_func: Function that returns value given spot
        spot: Current spot price
        bump_size: Relative bump size (e.g., 0.01 = 1%)
        method: "forward", "backward", or "central"

    Returns:
        Delta (sensitivity to spot)
    """
    bump = spot * bump_size

    if method == "central":
        v_up = pricing_func(spot + bump)
        v_down = pricing_func(spot - bump)
        delta = (v_up - v_down) / (2 * bump)
    elif method == "forward":
        v_base = pricing_func(spot)
        v_up = pricing_func(spot + bump)
        delta = (v_up - v_base) / bump
    else:  # backward
        v_base = pricing_func(spot)
        v_down = pricing_func(spot - bump)
        delta = (v_base - v_down) / bump

    return delta


def calculate_gamma(
    pricing_func: Callable[[float], float],
    spot: float,
    bump_size: float = 0.01,
) -> float:
    """
    Calculate gamma using finite difference.

    Gamma = (V(S+h) - 2V(S) + V(S-h)) / h²

    Args:
        pricing_func: Function that returns value given spot
        spot: Current spot price
        bump_size: Relative bump size

    Returns:
        Gamma (second derivative w.r.t. spot)
    """
    bump = spot * bump_size
    v_up = pricing_func(spot + bump)
    v_base = pricing_func(spot)
    v_down = pricing_func(spot - bump)

    return (v_up - 2 * v_base + v_down) / (bump ** 2)


def calculate_vega(
    pricing_func: Callable[[float], float],
    volatility: float,
    bump_size: float = 0.01,
) -> float:
    """
    Calculate vega (sensitivity to volatility).

    Args:
        pricing_func: Function that returns value given volatility
        volatility: Current volatility
        bump_size: Absolute bump (e.g., 0.01 = 1%)

    Returns:
        Vega (per 1% vol change)
    """
    v_up = pricing_func(volatility + bump_size)
    v_down = pricing_func(volatility - bump_size)

    return (v_up - v_down) / (2 * bump_size)


def calculate_theta(
    pricing_func: Callable[[date], float],
    valuation_date: date,
    days_forward: int = 1,
) -> float:
    """
    Calculate theta (time decay).

    Args:
        pricing_func: Function that returns value given valuation date
        valuation_date: Current valuation date
        days_forward: Number of days to advance

    Returns:
        Theta (per day)
    """
    v_today = pricing_func(valuation_date)
    v_tomorrow = pricing_func(valuation_date + timedelta(days=days_forward))

    return (v_tomorrow - v_today) / days_forward


def calculate_rho(
    pricing_func: Callable[[float], float],
    rate: float,
    bump_size: float = 0.0001,
) -> float:
    """
    Calculate rho (sensitivity to interest rate).

    Args:
        pricing_func: Function that returns value given rate
        rate: Current rate
        bump_size: Absolute bump (e.g., 0.0001 = 1bp)

    Returns:
        Rho (per 1bp)
    """
    v_up = pricing_func(rate + bump_size)
    v_down = pricing_func(rate - bump_size)

    return (v_up - v_down) / (2 * bump_size)


def full_greeks(
    price_func_spot: Callable[[float], float],
    price_func_vol: Callable[[float], float],
    price_func_time: Callable[[date], float],
    price_func_rate: Callable[[float], float],
    spot: float,
    volatility: float,
    valuation_date: date,
    rate: float,
    spot_bump: float = 0.01,
    vol_bump: float = 0.01,
    rate_bump: float = 0.0001,
) -> GreeksResult:
    """
    Calculate all first and second order Greeks.

    Args:
        price_func_spot: Pricing function of spot
        price_func_vol: Pricing function of volatility
        price_func_time: Pricing function of valuation date
        price_func_rate: Pricing function of rate
        spot: Current spot
        volatility: Current volatility
        valuation_date: Valuation date
        rate: Current rate
        spot_bump: Spot bump size (relative)
        vol_bump: Vol bump size (absolute)
        rate_bump: Rate bump size (absolute)

    Returns:
        GreeksResult
    """
    # First order
    delta = calculate_delta(price_func_spot, spot, spot_bump)
    gamma = calculate_gamma(price_func_spot, spot, spot_bump)
    vega = calculate_vega(price_func_vol, volatility, vol_bump)
    theta = calculate_theta(price_func_time, valuation_date)
    rho = calculate_rho(price_func_rate, rate, rate_bump)

    # Second order cross-Greeks (optional)
    # Vanna = ∂²V/∂S∂σ requires a joint pricing function of (spot, vol),
    # which the current API doesn't provide. Set to None.
    vanna = None

    try:
        # Volga = ∂²V/∂σ²
        v_up = price_func_vol(volatility + vol_bump)
        v_base = price_func_vol(volatility)
        v_down = price_func_vol(volatility - vol_bump)
        volga = (v_up - 2 * v_base + v_down) / (vol_bump ** 2)
    except (ArithmeticError, ValueError):
        volga = None

    return GreeksResult(
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
        vanna=vanna,
        volga=volga,
    )


def bucket_delta(
    pricing_func: Callable[[Curve], float],
    curve: Curve,
    bucket_tenors: List[float],
    bump_size: float = 1.0,
) -> Dict[float, float]:
    """
    Calculate bucket (key rate) delta.

    Bumps each tenor point individually to calculate sensitivity.

    Args:
        pricing_func: Function that returns value given curve
        curve: Yield curve
        bucket_tenors: Tenors for bucketing
        bump_size: Bump size in basis points

    Returns:
        Dict of {tenor: delta}
    """
    results = {}
    base_value = pricing_func(curve)

    for tenor in bucket_tenors:
        # Create curve with single tenor bump
        # This is a simplified implementation
        bumped_values = curve.values.copy()

        # Find nearest tenor
        idx = min(range(len(curve.tenors)), key=lambda i: abs(curve.tenors[i] - tenor))
        bumped_values[idx] += bump_size / 10000

        # Create bumped curve
        bumped_curve = type(curve)(
            reference_date=curve.reference_date,
            tenors=curve.tenors,
            values=bumped_values,
            interpolation=curve._interpolation_method,
            day_count=curve._day_count,
        )

        bumped_value = pricing_func(bumped_curve)
        results[tenor] = bumped_value - base_value

    return results


def key_rate_duration(
    pricing_func: Callable[[Curve], float],
    curve: Curve,
    key_rates: Optional[List[float]] = None,
    bump_size: float = 1.0,
) -> Dict[float, float]:
    """
    Calculate key rate durations.

    Key rate duration at tenor T measures sensitivity to
    a bump at that tenor with linear interpolation to
    adjacent tenors.

    Args:
        pricing_func: Function that returns value given curve
        curve: Yield curve
        key_rates: Tenors for key rates (default: standard set)
        bump_size: Bump size in basis points

    Returns:
        Dict of {tenor: duration}
    """
    if key_rates is None:
        key_rates = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]

    base_value = pricing_func(curve)
    results = {}

    for key_rate in key_rates:
        # Create key rate bump (triangular)
        bumped_values = curve.values.copy()

        for i, t in enumerate(curve.tenors):
            if t == key_rate:
                bump = bump_size / 10000
            elif t < key_rate:
                # Find previous key rate
                prev_keys = [k for k in key_rates if k < key_rate]
                if prev_keys:
                    prev_key = max(prev_keys)
                    if t >= prev_key:
                        bump = (t - prev_key) / (key_rate - prev_key) * bump_size / 10000
                    else:
                        bump = 0
                else:
                    bump = t / key_rate * bump_size / 10000
            else:
                # Find next key rate
                next_keys = [k for k in key_rates if k > key_rate]
                if next_keys:
                    next_key = min(next_keys)
                    if t <= next_key:
                        bump = (next_key - t) / (next_key - key_rate) * bump_size / 10000
                    else:
                        bump = 0
                else:
                    bump = 0

            bumped_values[i] += bump

        bumped_curve = type(curve)(
            reference_date=curve.reference_date,
            tenors=curve.tenors,
            values=bumped_values,
            interpolation=curve._interpolation_method,
            day_count=curve._day_count,
        )

        bumped_value = pricing_func(bumped_curve)

        # Duration = -ΔV/V / Δr
        if abs(base_value) > 1e-10:
            krd = -(bumped_value - base_value) / base_value / (bump_size / 10000)
        else:
            krd = 0

        results[key_rate] = krd

    return results


def parallel_shift_sensitivity(
    pricing_func: Callable[[Curve], float],
    curve: Curve,
    shift_sizes: Optional[List[float]] = None,
) -> Dict[float, float]:
    """
    Calculate sensitivity to parallel curve shifts.

    Args:
        pricing_func: Function that returns value given curve
        curve: Yield curve
        shift_sizes: List of shifts in bp (default: [-100, -50, -25, -10, -1, 1, 10, 25, 50, 100])

    Returns:
        Dict of {shift: pnl_impact}
    """
    if shift_sizes is None:
        shift_sizes = [-100, -50, -25, -10, -1, 1, 10, 25, 50, 100]

    base_value = pricing_func(curve)
    results = {}

    for shift in shift_sizes:
        shifted_curve = curve.shift(shift)
        shifted_value = pricing_func(shifted_curve)
        results[shift] = shifted_value - base_value

    return results


def twist_sensitivity(
    pricing_func: Callable[[Curve], float],
    curve: Curve,
    twist_pairs: Optional[List[Tuple[float, float]]] = None,
) -> Dict[Tuple[float, float], float]:
    """
    Calculate sensitivity to curve twists.

    Args:
        pricing_func: Pricing function
        curve: Yield curve
        twist_pairs: List of (short_shift_bp, long_shift_bp) pairs

    Returns:
        Dict of {(short, long): pnl_impact}
    """
    if twist_pairs is None:
        twist_pairs = [
            (-25, 25),  # Steepening
            (25, -25),  # Flattening
            (-50, 50),
            (50, -50),
        ]

    base_value = pricing_func(curve)
    results = {}

    for short_bp, long_bp in twist_pairs:
        twisted = curve.twist(short_bp, long_bp)
        twisted_value = pricing_func(twisted)
        results[(short_bp, long_bp)] = twisted_value - base_value

    return results


def cross_gamma(
    pricing_func: Callable[[float, float], float],
    x: float,
    y: float,
    bump_x: float = 0.01,
    bump_y: float = 0.01,
) -> float:
    """
    Calculate cross-gamma (∂²V/∂x∂y).

    Args:
        pricing_func: Function of two variables
        x: First variable value
        y: Second variable value
        bump_x: Bump size for x
        bump_y: Bump size for y

    Returns:
        Cross-gamma
    """
    v_pp = pricing_func(x + bump_x, y + bump_y)
    v_pm = pricing_func(x + bump_x, y - bump_y)
    v_mp = pricing_func(x - bump_x, y + bump_y)
    v_mm = pricing_func(x - bump_x, y - bump_y)

    return (v_pp - v_pm - v_mp + v_mm) / (4 * bump_x * bump_y)
