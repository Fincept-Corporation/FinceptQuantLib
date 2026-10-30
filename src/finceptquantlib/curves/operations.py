"""
Curve Operations

Non-destructive curve transformations:
- ShiftedCurve: Parallel shift in rate/discount space
- TwistedCurve: Pivot point twist
- ScaledCurve: Multiplicative scaling
- TimeShiftedCurve: Roll forward/backward in time
- ProxyCurve: Proxy to another curve with spread
- CompositeCurve: Combine multiple curves
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum, auto
from typing import Optional, Union, TYPE_CHECKING

from finceptquantlib.core.foundations.types import Rate, DateLike, to_date

if TYPE_CHECKING:
    from finceptquantlib.curves.curve import Curve


# =============================================================================
# Shift Types
# =============================================================================

class ShiftType(Enum):
    """Type of curve shift."""
    PARALLEL = auto()      # Uniform shift across all tenors
    TWIST = auto()         # Pivoted twist
    BUTTERFLY = auto()     # Butterfly (wings up/down, belly opposite)
    KEY_RATE = auto()      # Shift at specific tenor only


class ShiftSpace(Enum):
    """Space in which to apply shift."""
    ZERO_RATE = auto()     # Shift zero rates
    FORWARD_RATE = auto()  # Shift forward rates
    DISCOUNT = auto()      # Shift discount factors


# =============================================================================
# Base Curve Wrapper
# =============================================================================

class CurveWrapper(ABC):
    """
    Abstract base for curve wrappers/transformations.

    Provides a common interface for transformed curves that
    delegate to an underlying base curve.
    """

    @property
    @abstractmethod
    def base_curve(self) -> "Curve":
        """The underlying base curve."""

    @property
    def reference_date(self) -> date:
        """Reference (valuation) date of the underlying curve."""
        return self.base_curve.reference_date

    def _resolve(self, dt: Union[float, DateLike]) -> tuple[Union[float, date], float]:
        """Split a query point into the value passed to the base curve and its time in years.

        Like ``Curve``, a number is a tenor in years from the reference date; anything
        else is converted to a date.
        """
        if isinstance(dt, (int, float)):
            return float(dt), float(dt)
        d = to_date(dt)
        return d, (d - self.reference_date).days / 365.0

    @abstractmethod
    def discount_factor(self, dt: Union[float, DateLike]) -> float:
        """Calculate discount factor at a tenor (years) or date."""

    @abstractmethod
    def zero_rate(self, dt: Union[float, DateLike]) -> float:
        """Calculate zero rate at a tenor (years) or date."""

    def forward_rate(
        self,
        start: Union[float, DateLike],
        end: Union[float, DateLike]
    ) -> float:
        """Calculate the simple forward rate between two tenors or dates."""
        start_x, start_t = self._resolve(start)
        end_x, end_t = self._resolve(end)

        t = end_t - start_t
        if t <= 0:
            return 0.0

        df_start = self.discount_factor(start_x)
        df_end = self.discount_factor(end_x)
        return (df_start / df_end - 1) / t


# =============================================================================
# Shifted Curve
# =============================================================================

@dataclass
class ShiftedCurve(CurveWrapper):
    """
    Curve with parallel or shaped shift applied.

    Supports shifting in zero rate, forward rate, or discount factor space.

    Examples:
        >>> shifted = ShiftedCurve(base, shift=0.001)  # +10bps parallel  # doctest: +SKIP
        >>> shifted = ShiftedCurve(base, shift=Rate("10bps"))  # doctest: +SKIP
        >>> shifted = ShiftedCurve(base, shift=-0.0025, space=ShiftSpace.FORWARD_RATE)  # doctest: +SKIP
    """
    _base: "Curve"
    shift: Union[float, Rate]
    space: ShiftSpace = ShiftSpace.ZERO_RATE
    shift_type: ShiftType = ShiftType.PARALLEL

    def __post_init__(self) -> None:
        if isinstance(self.shift, Rate):
            self.shift = self.shift.decimal

    @property
    def base_curve(self) -> "Curve":
        return self._base

    def discount_factor(self, dt: Union[float, DateLike]) -> float:
        """Calculate shifted discount factor."""
        dt, t = self._resolve(dt)

        if t <= 0:
            return 1.0

        if self.space == ShiftSpace.DISCOUNT:
            # Multiplicative shift to discount factors
            base_df = self._base.discount_factor(dt)
            return base_df * math.exp(-self.shift * t)

        elif self.space == ShiftSpace.ZERO_RATE:
            # Shift zero rates, recalculate DF
            base_zero = self._base.zero_rate(dt)
            shifted_zero = base_zero + self.shift
            return math.exp(-shifted_zero * t)

        elif self.space == ShiftSpace.FORWARD_RATE:
            # Shift forward rates - more complex
            # For parallel shift, equivalent to zero rate shift
            base_zero = self._base.zero_rate(dt)
            shifted_zero = base_zero + self.shift
            return math.exp(-shifted_zero * t)

        return self._base.discount_factor(dt)

    def zero_rate(self, dt: Union[float, DateLike]) -> float:
        """Calculate shifted zero rate."""
        dt, t = self._resolve(dt)

        if self.space == ShiftSpace.ZERO_RATE:
            return self._base.zero_rate(dt) + self.shift

        # For other spaces, derive from discount factor
        df = self.discount_factor(dt)
        if t <= 0:
            return self._base.zero_rate(dt)
        return -math.log(df) / t

    def shift_amount(self) -> float:
        """Get the shift amount in decimal."""
        return self.shift

    def __repr__(self) -> str:
        shift_bps = self.shift * 10000
        return f"ShiftedCurve({shift_bps:+.1f}bps, {self.space.name})"


# =============================================================================
# Twisted Curve
# =============================================================================

@dataclass
class TwistedCurve(CurveWrapper):
    """
    Curve with pivot twist applied.

    Twist = rotation around a pivot point.
    Points before pivot shift one direction, after shift opposite.

    Args:
        _base: Base curve
        pivot_tenor: Pivot point tenor (e.g., "5Y")
        short_shift: Shift for short end (before pivot)
        long_shift: Shift for long end (after pivot)

    The shift at any point is linearly interpolated from the pivot.
    """
    _base: "Curve"
    pivot_date: date
    short_shift: float  # Shift at short end (0Y)
    long_shift: float   # Shift at long end (30Y default)
    long_end_years: float = 30.0

    @property
    def base_curve(self) -> "Curve":
        return self._base

    def _get_shift(self, t: float) -> float:
        """Calculate shift amount at time t (years)."""
        pivot_t = (self.pivot_date - self.reference_date).days / 365.0

        if t <= 0:
            return self.short_shift

        if t <= pivot_t:
            # Interpolate between short end and pivot (zero shift at pivot)
            weight = t / pivot_t if pivot_t > 0 else 0
            return self.short_shift * (1 - weight)
        else:
            # Interpolate between pivot and long end
            remaining = self.long_end_years - pivot_t
            if remaining <= 0:
                return self.long_shift
            weight = (t - pivot_t) / remaining
            weight = min(weight, 1.0)
            return self.long_shift * weight

    def discount_factor(self, dt: Union[float, DateLike]) -> float:
        """Calculate twisted discount factor."""
        dt, t = self._resolve(dt)

        if t <= 0:
            return 1.0

        shift = self._get_shift(t)
        base_zero = self._base.zero_rate(dt)
        return math.exp(-(base_zero + shift) * t)

    def zero_rate(self, dt: Union[float, DateLike]) -> float:
        """Calculate twisted zero rate."""
        dt, t = self._resolve(dt)
        return self._base.zero_rate(dt) + self._get_shift(t)

    def __repr__(self) -> str:
        short_bps = self.short_shift * 10000
        long_bps = self.long_shift * 10000
        return f"TwistedCurve(short={short_bps:+.1f}bps, long={long_bps:+.1f}bps)"


# =============================================================================
# Butterfly Shifted Curve
# =============================================================================

@dataclass
class ButterflyShiftedCurve(CurveWrapper):
    """
    Curve with butterfly shift applied.

    Butterfly = wings move opposite to belly.
    Short and long ends shift one direction, belly shifts opposite.

    Args:
        _base: Base curve
        belly_date: Belly point date
        wing_shift: Shift for wings (short and long)
        belly_shift: Shift for belly (typically opposite sign)
    """
    _base: "Curve"
    belly_date: date
    wing_shift: float
    belly_shift: float
    short_end_years: float = 0.0
    long_end_years: float = 30.0

    @property
    def base_curve(self) -> "Curve":
        return self._base

    def _get_shift(self, t: float) -> float:
        """Calculate butterfly shift at time t (years)."""
        belly_t = (self.belly_date - self.reference_date).days / 365.0

        if t <= self.short_end_years:
            return self.wing_shift

        if t >= self.long_end_years:
            return self.wing_shift

        # Calculate position relative to belly
        if t <= belly_t:
            # Short wing to belly
            if belly_t - self.short_end_years <= 0:
                return self.belly_shift
            weight = (t - self.short_end_years) / (belly_t - self.short_end_years)
            return self.wing_shift + (self.belly_shift - self.wing_shift) * weight
        else:
            # Belly to long wing
            if self.long_end_years - belly_t <= 0:
                return self.wing_shift
            weight = (t - belly_t) / (self.long_end_years - belly_t)
            return self.belly_shift + (self.wing_shift - self.belly_shift) * weight

    def discount_factor(self, dt: Union[float, DateLike]) -> float:
        """Calculate butterfly-shifted discount factor."""
        dt, t = self._resolve(dt)

        if t <= 0:
            return 1.0

        shift = self._get_shift(t)
        base_zero = self._base.zero_rate(dt)
        return math.exp(-(base_zero + shift) * t)

    def zero_rate(self, dt: Union[float, DateLike]) -> float:
        """Calculate butterfly-shifted zero rate."""
        dt, t = self._resolve(dt)
        return self._base.zero_rate(dt) + self._get_shift(t)

    def __repr__(self) -> str:
        wing_bps = self.wing_shift * 10000
        belly_bps = self.belly_shift * 10000
        return f"ButterflyShiftedCurve(wings={wing_bps:+.1f}bps, belly={belly_bps:+.1f}bps)"


# =============================================================================
# Time-Shifted (Rolled) Curve
# =============================================================================

@dataclass
class TimeShiftedCurve(CurveWrapper):
    """
    Curve rolled forward or backward in time.

    Useful for:
    - Theta/carry calculations
    - What-if analysis for future dates
    - Historical curve comparisons

    Args:
        _base: Base curve
        days: Number of days to roll (positive = forward)
        preserve_shape: If True, maintain curve shape; if False, slide along curve
    """
    _base: "Curve"
    days: int
    preserve_shape: bool = True

    @property
    def base_curve(self) -> "Curve":
        return self._base

    @property
    def reference_date(self) -> date:
        """Rolled reference date."""
        return self._base.reference_date + timedelta(days=self.days)

    def discount_factor(self, dt: Union[float, DateLike]) -> float:
        """Calculate discount factor on rolled curve (from the rolled reference date)."""
        _, t = self._resolve(dt)
        if t <= 0:
            return 1.0
        return math.exp(-self.zero_rate(t) * t)

    def zero_rate(self, dt: Union[float, DateLike]) -> float:
        """Calculate zero rate on rolled curve."""
        _, t = self._resolve(dt)

        if self.preserve_shape:
            # Same rate at the same time to maturity as the base curve
            return self._base.zero_rate(max(t, 0.0))
        # Slide along the curve: the rate stays attached to the calendar date,
        # which is t + days/365 years after the base curve's reference date
        return self._base.zero_rate(t + self.days / 365.0)

    def __repr__(self) -> str:
        direction = "forward" if self.days > 0 else "backward"
        return f"TimeShiftedCurve({abs(self.days)} days {direction})"


# =============================================================================
# Scaled Curve
# =============================================================================

@dataclass
class ScaledCurve(CurveWrapper):
    """
    Curve with rates scaled by a factor.

    Useful for:
    - Stress testing (e.g., rates * 1.5)
    - Scenario analysis

    Args:
        _base: Base curve
        scale_factor: Multiplicative scale for rates
    """
    _base: "Curve"
    scale_factor: float

    @property
    def base_curve(self) -> "Curve":
        return self._base

    def discount_factor(self, dt: Union[float, DateLike]) -> float:
        """Calculate scaled discount factor."""
        dt, t = self._resolve(dt)

        if t <= 0:
            return 1.0

        base_zero = self._base.zero_rate(dt)
        scaled_zero = base_zero * self.scale_factor
        return math.exp(-scaled_zero * t)

    def zero_rate(self, dt: Union[float, DateLike]) -> float:
        """Calculate scaled zero rate."""
        dt, _ = self._resolve(dt)
        return self._base.zero_rate(dt) * self.scale_factor

    def __repr__(self) -> str:
        return f"ScaledCurve(x{self.scale_factor:.2f})"


# =============================================================================
# Proxy Curve
# =============================================================================

@dataclass
class ProxyCurve(CurveWrapper):
    """
    Proxy curve that delegates to another with a spread.

    Useful for:
    - Credit-adjusted discounting
    - Cross-currency basis
    - Approximating one curve from another

    Args:
        _base: Base reference curve
        spread: Spread to add to base rates
        spread_curve: Optional curve for spread (vs constant)
    """
    _base: "Curve"
    spread: Union[float, Rate] = 0.0
    spread_curve: Optional["Curve"] = None

    def __post_init__(self) -> None:
        if isinstance(self.spread, Rate):
            self.spread = self.spread.decimal

    @property
    def base_curve(self) -> "Curve":
        return self._base

    def _get_spread(self, dt: Union[float, date]) -> float:
        """Get spread at a tenor (years) or date."""
        if self.spread_curve is not None:
            return self.spread_curve.zero_rate(dt)
        return self.spread

    def discount_factor(self, dt: Union[float, DateLike]) -> float:
        """Calculate proxied discount factor."""
        dt, t = self._resolve(dt)

        if t <= 0:
            return 1.0

        base_zero = self._base.zero_rate(dt)
        spread = self._get_spread(dt)
        return math.exp(-(base_zero + spread) * t)

    def zero_rate(self, dt: Union[float, DateLike]) -> float:
        """Calculate proxied zero rate."""
        dt, _ = self._resolve(dt)
        return self._base.zero_rate(dt) + self._get_spread(dt)

    def __repr__(self) -> str:
        if self.spread_curve:
            return "ProxyCurve(+spread_curve)"
        spread_bps = self.spread * 10000
        return f"ProxyCurve({spread_bps:+.1f}bps)"


# =============================================================================
# Composite Curve
# =============================================================================

@dataclass
class CompositeCurve(CurveWrapper):
    """
    Curve combining discount and forecast curves.

    Used for dual-curve pricing where:
    - Discounting uses one curve (e.g., OIS)
    - Forecasting uses another (e.g., LIBOR/SOFR)

    Args:
        discount_curve: Curve for discounting cashflows
        forecast_curve: Curve for projecting floating rates
    """
    discount_curve: "Curve"
    forecast_curve: "Curve"

    @property
    def base_curve(self) -> "Curve":
        return self.discount_curve

    def discount_factor(self, dt: Union[float, DateLike]) -> float:
        """Get discount factor from discount curve."""
        dt, _ = self._resolve(dt)
        return self.discount_curve.discount_factor(dt)

    def zero_rate(self, dt: Union[float, DateLike]) -> float:
        """Get zero rate from discount curve."""
        dt, _ = self._resolve(dt)
        return self.discount_curve.zero_rate(dt)

    def forecast_rate(self, start: Union[float, DateLike], end: Union[float, DateLike]) -> float:
        """Get forward rate from forecast curve."""
        return self.forward_rate(start, end)

    def forward_rate(self, start: Union[float, DateLike], end: Union[float, DateLike]) -> float:
        """Forward rate from forecast curve (for floating rate projection)."""
        start, _ = self._resolve(start)
        end, _ = self._resolve(end)
        return self.forecast_curve.forward_rate(start, end)

    def __repr__(self) -> str:
        return f"CompositeCurve(discount={self.discount_curve}, forecast={self.forecast_curve})"


# =============================================================================
# Key Rate Shifted Curve
# =============================================================================

@dataclass
class KeyRateShiftedCurve(CurveWrapper):
    """
    Curve with key rate duration shift.

    Applies a shift at specific tenor points that tapers off
    to neighboring tenors.

    Args:
        _base: Base curve
        key_tenor_years: Tenor point to shift (in years)
        shift: Shift amount at key tenor
        width_years: Width of shift influence (each side)
    """
    _base: "Curve"
    key_tenor_years: float
    shift: float
    width_years: float = 2.0

    @property
    def base_curve(self) -> "Curve":
        return self._base

    def _get_shift(self, t: float) -> float:
        """Calculate shift at time t (years) with a triangular kernel centred at the key tenor."""

        distance = abs(t - self.key_tenor_years)
        if distance >= self.width_years:
            return 0.0

        # Triangular taper
        weight = 1 - distance / self.width_years
        return self.shift * weight

    def discount_factor(self, dt: Union[float, DateLike]) -> float:
        """Calculate key-rate shifted discount factor."""
        dt, t = self._resolve(dt)

        if t <= 0:
            return 1.0

        shift = self._get_shift(t)
        base_zero = self._base.zero_rate(dt)
        return math.exp(-(base_zero + shift) * t)

    def zero_rate(self, dt: Union[float, DateLike]) -> float:
        """Calculate key-rate shifted zero rate."""
        dt, t = self._resolve(dt)
        return self._base.zero_rate(dt) + self._get_shift(t)

    def __repr__(self) -> str:
        shift_bps = self.shift * 10000
        return f"KeyRateShiftedCurve({self.key_tenor_years}Y: {shift_bps:+.1f}bps)"


# =============================================================================
# Convenience Functions
# =============================================================================

def parallel_shift(
    curve: "Curve",
    shift_bps: float,
    space: ShiftSpace = ShiftSpace.ZERO_RATE
) -> ShiftedCurve:
    """Create a parallel-shifted curve."""
    return ShiftedCurve(
        _base=curve,
        shift=shift_bps / 10000.0,
        space=space
    )


def twist_curve(
    curve: "Curve",
    pivot_years: float,
    short_shift_bps: float,
    long_shift_bps: float
) -> TwistedCurve:
    """Create a twisted curve."""
    pivot_date = curve.reference_date + timedelta(days=int(pivot_years * 365))
    return TwistedCurve(
        _base=curve,
        pivot_date=pivot_date,
        short_shift=short_shift_bps / 10000.0,
        long_shift=long_shift_bps / 10000.0
    )


def butterfly_shift(
    curve: "Curve",
    belly_years: float,
    wing_shift_bps: float,
    belly_shift_bps: float
) -> ButterflyShiftedCurve:
    """Create a butterfly-shifted curve."""
    belly_date = curve.reference_date + timedelta(days=int(belly_years * 365))
    return ButterflyShiftedCurve(
        _base=curve,
        belly_date=belly_date,
        wing_shift=wing_shift_bps / 10000.0,
        belly_shift=belly_shift_bps / 10000.0
    )


def roll_curve(
    curve: "Curve",
    days: int,
    preserve_shape: bool = True
) -> TimeShiftedCurve:
    """Create a time-rolled curve."""
    return TimeShiftedCurve(
        _base=curve,
        days=days,
        preserve_shape=preserve_shape
    )


def key_rate_shift(
    curve: "Curve",
    tenor_years: float,
    shift_bps: float,
    width_years: float = 2.0
) -> KeyRateShiftedCurve:
    """Create a key-rate shifted curve."""
    return KeyRateShiftedCurve(
        _base=curve,
        key_tenor_years=tenor_years,
        shift=shift_bps / 10000.0,
        width_years=width_years
    )
