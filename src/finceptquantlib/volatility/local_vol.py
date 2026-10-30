"""
Local Volatility Models - Base Classes

Provides base classes and simple implementations:
- LocalVolatility: Abstract base class
- ConstantLocalVol: Constant vol (Black-Scholes case)
- ParametricLocalVol: Parametric local vol with level and time effects
- implied_to_local_vol: Convert implied vol to local vol at a point

For comprehensive local vol implementation, see finceptquantlib.models.local_vol
which provides DupireLocalVol, LocalVolSurface, SVI, and SSVI classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Callable
import math

from finceptquantlib.core.utilities.distributions import norm_cdf


class LocalVolatility(ABC):
    """Abstract base for local volatility models."""

    @abstractmethod
    def local_vol(self, spot: float, time: float) -> float:
        """Get local volatility at (S, t)."""
        pass


class ConstantLocalVol(LocalVolatility):
    """
    Constant local volatility (Black-Scholes case).

    In this model, local vol equals implied vol everywhere,
    and both are constant regardless of strike or time.
    """

    def __init__(self, volatility: float):
        """
        Initialize constant local vol model.

        Args:
            volatility: Constant volatility level
        """
        if volatility < 0:
            raise ValueError("Volatility must be non-negative")
        self._vol = volatility

    def local_vol(self, spot: float, time: float) -> float:
        """Get local volatility (constant)."""
        return self._vol


class ParametricLocalVol(LocalVolatility):
    """
    Parametric local volatility model.

    σ(S, t) = σ₀ * (S/S₀)^β * f(t)

    where:
    - σ₀ is the base (ATM) volatility
    - β is the CEV-like skew parameter
    - f(t) is a time-dependent term structure factor

    Parameters:
    - β < 0: Volatility increases as spot decreases (leverage effect)
    - β = 0: No spot dependence (pure term structure)
    - β > 0: Volatility increases as spot increases

    Example:
        >>> # Equity-like with leverage effect
        >>> model = ParametricLocalVol(spot=100, base_vol=0.20, beta=-0.5)
        >>> model.local_vol(90, 0.5)   # Higher vol at lower spot
        0.2108...
        >>> model.local_vol(110, 0.5)  # Lower vol at higher spot
        0.1906...
    """

    def __init__(
        self,
        spot: float,
        base_vol: float,
        beta: float = 0.0,
        term_structure: Optional[Callable[[float], float]] = None,
    ):
        """
        Initialize parametric local vol model.

        Args:
            spot: Reference spot price (S₀)
            base_vol: Base volatility at reference spot
            beta: CEV-like skew parameter
            term_structure: Function f(t) for time dependence (default: constant 1.0)
        """
        if spot <= 0:
            raise ValueError("Spot must be positive")
        if base_vol < 0:
            raise ValueError("Base volatility must be non-negative")

        self._spot = spot
        self._base_vol = base_vol
        self._beta = beta
        self._term_structure = term_structure or (lambda t: 1.0)

    def local_vol(self, spot: float, time: float) -> float:
        """
        Get local volatility at (S, t).

        Args:
            spot: Current spot price
            time: Time point

        Returns:
            Local volatility
        """
        if spot <= 0:
            return self._base_vol

        level_effect = (spot / self._spot) ** self._beta
        time_effect = self._term_structure(time)
        return self._base_vol * level_effect * time_effect


def implied_to_local_vol(
    implied_vol_func: Callable[[float, float], float],
    spot: float,
    rate: float,
    strike: float,
    expiry: float,
    dk: float = 0.01,
    dt: float = 0.01,
) -> float:
    """
    Convert implied volatility to local volatility at a point.

    Uses Dupire formula with numerical derivatives:

    σ²_loc(K,T) = [∂C/∂T + rK·∂C/∂K] / [½K²·∂²C/∂K²]

    where C(K,T) is the undiscounted call price.

    Args:
        implied_vol_func: Function sigma_impl(K, T) returning implied volatility
        spot: Current spot price
        rate: Risk-free rate
        strike: Strike price
        expiry: Time to expiry
        dk: Strike bump fraction for numerical derivative (default 1%)
        dt: Time bump for numerical derivative (default 0.01)

    Returns:
        Local volatility at (K, T)

    Example:
        >>> # Flat implied vol surface
        >>> flat_vol = lambda K, T: 0.20
        >>> # Local vol equals implied vol for a flat surface
        >>> round(implied_to_local_vol(flat_vol, spot=100, rate=0.05, strike=100, expiry=1.0), 3)
        0.2
    """
    def call_price(k: float, t: float) -> float:
        """Compute Black-Scholes call price."""
        fwd = spot * math.exp(rate * t)
        if t < 1e-6:
            return max(fwd - k, 0)
        vol = implied_vol_func(k, t)
        if vol <= 0:
            return max(fwd - k, 0)
        sqrt_t = math.sqrt(t)
        d1 = (math.log(fwd / k) + 0.5 * vol ** 2 * t) / (vol * sqrt_t)
        d2 = d1 - vol * sqrt_t
        df = math.exp(-rate * t)
        return df * (fwd * norm_cdf(d1) - k * norm_cdf(d2))

    # Numerical derivatives
    c = call_price(strike, expiry)
    c_up_t = call_price(strike, expiry + dt)
    c_up_k = call_price(strike + dk * strike, expiry)
    c_down_k = call_price(max(strike - dk * strike, 1e-6), expiry)

    if expiry - dt > 1e-6:
        # Central difference: second-order accurate in dt
        dcdt = (c_up_t - call_price(strike, expiry - dt)) / (2 * dt)
    else:
        dcdt = (c_up_t - c) / dt
    dcdk = (c_up_k - c_down_k) / (2 * dk * strike)
    d2cdk2 = (c_up_k - 2 * c + c_down_k) / (dk * strike) ** 2

    # Dupire formula
    numerator = dcdt + rate * strike * dcdk
    denominator = 0.5 * strike ** 2 * d2cdk2

    if denominator <= 0:
        # Numerical issue or arbitrage - fall back to implied vol
        return implied_vol_func(strike, expiry)

    local_var = numerator / denominator

    if local_var < 0:
        # Negative variance - fall back to implied vol
        return implied_vol_func(strike, expiry)

    return math.sqrt(local_var)
