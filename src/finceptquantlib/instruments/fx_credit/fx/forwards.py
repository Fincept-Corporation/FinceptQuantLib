"""
FX Forwards and Swaps

Implements:
- FX Spot trades
- FX Forwards (outrights)
- FX Swaps (spot + forward)
- Forward points calculation

Key Concepts:
- Covered Interest Rate Parity (CIRP)
- Forward rate: F = S * exp((r_d - r_f) * T)
- Base/Quote convention (e.g., EUR/USD means EUR is base)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, Tuple
import math

from finceptquantlib.curves.curve import Curve
from finceptquantlib.scheduling.calendar import Calendar
from finceptquantlib.scheduling.adjustment import BusinessDayAdjustment


@dataclass
class FXForwardResult:
    """Result of FX forward valuation."""
    present_value_base: float     # PV in base currency
    present_value_quote: float    # PV in quote currency
    forward_rate: float           # Implied forward rate
    forward_points: float         # Forward - Spot (in pips)
    delta_base: float             # Sensitivity in base currency
    delta_quote: float            # Sensitivity in quote currency


class FXSpot:
    """
    FX Spot Transaction.

    Exchange of currencies at spot rate with T+2 settlement.
    """

    def __init__(
        self,
        base_currency: str,
        quote_currency: str,
        base_amount: float,
        spot_rate: float,
        trade_date: date,
        settlement_days: int = 2,
        calendar: Optional[Calendar] = None,
    ):
        """
        Initialize FX spot trade.

        Args:
            base_currency: Base currency (e.g., "EUR")
            quote_currency: Quote currency (e.g., "USD")
            base_amount: Amount in base currency (positive = buy, negative = sell)
            spot_rate: Spot FX rate (quote per base)
            trade_date: Trade date
            settlement_days: Days to settlement (typically 2)
            calendar: Business day calendar
        """
        self.base_currency = base_currency.upper()
        self.quote_currency = quote_currency.upper()
        self.base_amount = base_amount
        self.spot_rate = spot_rate
        self.trade_date = trade_date
        self.settlement_days = settlement_days
        self.calendar = calendar

        # Calculate settlement date
        self._settlement_date = self._calculate_settlement()

    @property
    def pair(self) -> str:
        """Currency pair string."""
        return f"{self.base_currency}/{self.quote_currency}"

    @property
    def quote_amount(self) -> float:
        """Amount in quote currency."""
        return -self.base_amount * self.spot_rate

    @property
    def settlement_date(self) -> date:
        """Settlement date."""
        return self._settlement_date

    def _calculate_settlement(self) -> date:
        """Calculate settlement date."""
        settle = self.trade_date + timedelta(days=self.settlement_days)
        if self.calendar:
            adj = BusinessDayAdjustment("following", self.calendar)
            settle = adj.adjust(settle)
        return settle

    def present_value(
        self,
        valuation_date: date,
        base_curve: Curve,
        quote_curve: Curve,
        current_spot: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Calculate present value.

        Args:
            valuation_date: Valuation date
            base_curve: Discount curve for base currency
            quote_curve: Discount curve for quote currency
            current_spot: Current spot rate (uses original if None)

        Returns:
            (PV in base currency, PV in quote currency)
        """
        if current_spot is None:
            current_spot = self.spot_rate

        if self._settlement_date <= valuation_date:
            return (0.0, 0.0)

        # Discount factors
        df_base = base_curve.discount_factor(self._settlement_date)
        df_quote = quote_curve.discount_factor(self._settlement_date)

        # PV of receiving base and paying quote
        pv_base = self.base_amount * df_base
        pv_quote = self.quote_amount * df_quote

        # Convert quote to base for total PV in base
        pv_in_base = pv_base + pv_quote / current_spot

        return (pv_in_base, pv_in_base * current_spot)

    def __repr__(self) -> str:
        direction = "Buy" if self.base_amount > 0 else "Sell"
        return (f"FXSpot({direction} {abs(self.base_amount):,.2f} {self.pair} "
                f"@ {self.spot_rate:.5f})")


class FXForward:
    """
    FX Forward (Outright).

    Agreement to exchange currencies at a future date at a predetermined rate.

    The forward rate is determined by interest rate parity:
    F = S * exp((r_quote - r_base) * T)

    Or using discrete compounding:
    F = S * (1 + r_quote * T) / (1 + r_base * T)
    """

    def __init__(
        self,
        base_currency: str,
        quote_currency: str,
        base_amount: float,
        forward_rate: float,
        value_date: date,
        trade_date: Optional[date] = None,
        spot_rate: Optional[float] = None,
    ):
        """
        Initialize FX forward.

        Args:
            base_currency: Base currency
            quote_currency: Quote currency
            base_amount: Amount in base currency
            forward_rate: Agreed forward rate
            value_date: Settlement/value date
            trade_date: Trade date (optional)
            spot_rate: Spot rate at trade (for forward points calculation)
        """
        self.base_currency = base_currency.upper()
        self.quote_currency = quote_currency.upper()
        self.base_amount = base_amount
        self.forward_rate = forward_rate
        self.value_date = value_date
        self.trade_date = trade_date or date.today()
        self.spot_rate = spot_rate

    @property
    def pair(self) -> str:
        """Currency pair string."""
        return f"{self.base_currency}/{self.quote_currency}"

    @property
    def quote_amount(self) -> float:
        """Amount in quote currency."""
        return -self.base_amount * self.forward_rate

    @property
    def forward_points(self) -> Optional[float]:
        """Forward points (forward - spot) in pips."""
        if self.spot_rate is None:
            return None
        return (self.forward_rate - self.spot_rate) * 10000

    @classmethod
    def from_spot(
        cls,
        base_currency: str,
        quote_currency: str,
        base_amount: float,
        spot_rate: float,
        value_date: date,
        base_curve: Curve,
        quote_curve: Curve,
        trade_date: Optional[date] = None,
    ) -> "FXForward":
        """
        Create forward from spot rate and curves.

        Uses covered interest rate parity to calculate forward rate.

        Args:
            base_currency: Base currency
            quote_currency: Quote currency
            base_amount: Amount in base currency
            spot_rate: Current spot rate
            value_date: Forward value date
            base_curve: Base currency discount curve
            quote_curve: Quote currency discount curve
            trade_date: Trade date

        Returns:
            FXForward
        """
        trade_date = trade_date or date.today()

        # Forward rate from interest rate parity
        # F = S * DF_base / DF_quote
        df_base = base_curve.discount_factor(value_date)
        df_quote = quote_curve.discount_factor(value_date)

        forward_rate = spot_rate * df_base / df_quote

        return cls(
            base_currency=base_currency,
            quote_currency=quote_currency,
            base_amount=base_amount,
            forward_rate=forward_rate,
            value_date=value_date,
            trade_date=trade_date,
            spot_rate=spot_rate,
        )

    def present_value(
        self,
        valuation_date: date,
        base_curve: Curve,
        quote_curve: Curve,
        current_spot: float,
    ) -> Tuple[float, float]:
        """
        Calculate present value.

        PV_base = base_amount * DF_base - quote_amount * DF_quote / current_spot

        Returns:
            (PV in base currency, PV in quote currency)
        """
        if self.value_date <= valuation_date:
            return (0.0, 0.0)

        df_base = base_curve.discount_factor(self.value_date)
        df_quote = quote_curve.discount_factor(self.value_date)

        # PV of base leg
        pv_base_leg = self.base_amount * df_base

        # PV of quote leg (paying quote_amount)
        pv_quote_leg = self.quote_amount * df_quote

        # Total PV in base terms
        pv_base = pv_base_leg + pv_quote_leg / current_spot
        pv_quote = pv_base * current_spot

        return (pv_base, pv_quote)

    def market_value(
        self,
        valuation_date: date,
        base_curve: Curve,
        quote_curve: Curve,
        current_spot: float,
    ) -> float:
        """
        Calculate mark-to-market value.

        MTM = Notional * (F_market - F_contract) * DF

        Returns:
            MTM value in quote currency
        """
        if self.value_date <= valuation_date:
            return 0.0

        # Calculate current market forward
        df_base = base_curve.discount_factor(self.value_date)
        df_quote = quote_curve.discount_factor(self.value_date)
        market_forward = current_spot * df_base / df_quote

        # MTM
        mtm = self.base_amount * (market_forward - self.forward_rate) * df_quote

        return mtm

    def value(
        self,
        valuation_date: date,
        base_curve: Curve,
        quote_curve: Curve,
        current_spot: float,
    ) -> FXForwardResult:
        """Full valuation with all analytics."""
        pv_base, pv_quote = self.present_value(
            valuation_date, base_curve, quote_curve, current_spot
        )

        # Calculate market forward
        df_base = base_curve.discount_factor(self.value_date)
        df_quote = quote_curve.discount_factor(self.value_date)
        market_forward = current_spot * df_base / df_quote

        # Forward points
        fwd_points = (market_forward - current_spot) * 10000

        # Delta (sensitivity to spot)
        bump = 0.0001  # 1 pip
        pv_up_base, _ = self.present_value(
            valuation_date, base_curve, quote_curve, current_spot + bump
        )
        delta_base = (pv_up_base - pv_base) / bump
        delta_quote = delta_base * current_spot

        return FXForwardResult(
            present_value_base=pv_base,
            present_value_quote=pv_quote,
            forward_rate=market_forward,
            forward_points=fwd_points,
            delta_base=delta_base,
            delta_quote=delta_quote,
        )

    def __repr__(self) -> str:
        direction = "Buy" if self.base_amount > 0 else "Sell"
        return (f"FXForward({direction} {abs(self.base_amount):,.2f} {self.pair} "
                f"@ {self.forward_rate:.5f} for {self.value_date})")


class FXSwap:
    """
    FX Swap.

    Combination of a spot and forward trade in opposite directions.
    Used for rolling positions or funding in foreign currency.

    Near leg: Exchange currencies at spot
    Far leg: Reverse exchange at forward rate
    """

    def __init__(
        self,
        base_currency: str,
        quote_currency: str,
        base_amount_near: float,
        spot_rate: float,
        forward_rate: float,
        near_date: date,
        far_date: date,
        swap_points: Optional[float] = None,
    ):
        """
        Initialize FX swap.

        Args:
            base_currency: Base currency
            quote_currency: Quote currency
            base_amount_near: Base amount on near leg (positive = buy)
            spot_rate: Spot rate for near leg
            forward_rate: Forward rate for far leg
            near_date: Near leg settlement date
            far_date: Far leg settlement date
            swap_points: Forward points (optional)
        """
        self.base_currency = base_currency.upper()
        self.quote_currency = quote_currency.upper()
        self.base_amount_near = base_amount_near
        self.spot_rate = spot_rate
        self.forward_rate = forward_rate
        self.near_date = near_date
        self.far_date = far_date

        if swap_points is None:
            self.swap_points = (forward_rate - spot_rate) * 10000
        else:
            self.swap_points = swap_points

        # Create component legs
        self.near_leg = FXSpot(
            base_currency=base_currency,
            quote_currency=quote_currency,
            base_amount=base_amount_near,
            spot_rate=spot_rate,
            trade_date=near_date - timedelta(days=2),
        )
        self.near_leg._settlement_date = near_date

        self.far_leg = FXForward(
            base_currency=base_currency,
            quote_currency=quote_currency,
            base_amount=-base_amount_near,  # Opposite direction
            forward_rate=forward_rate,
            value_date=far_date,
            spot_rate=spot_rate,
        )

    @property
    def pair(self) -> str:
        """Currency pair string."""
        return f"{self.base_currency}/{self.quote_currency}"

    @classmethod
    def from_curves(
        cls,
        base_currency: str,
        quote_currency: str,
        base_amount: float,
        spot_rate: float,
        near_date: date,
        far_date: date,
        base_curve: Curve,
        quote_curve: Curve,
    ) -> "FXSwap":
        """
        Create FX swap with forward rate derived from curves.

        Args:
            base_currency: Base currency
            quote_currency: Quote currency
            base_amount: Base amount on near leg
            spot_rate: Spot rate
            near_date: Near leg date
            far_date: Far leg date
            base_curve: Base currency curve
            quote_curve: Quote currency curve

        Returns:
            FXSwap
        """
        # Calculate forward rate
        df_base = base_curve.discount_factor(far_date) / base_curve.discount_factor(near_date)
        df_quote = quote_curve.discount_factor(far_date) / quote_curve.discount_factor(near_date)

        forward_rate = spot_rate * df_base / df_quote

        return cls(
            base_currency=base_currency,
            quote_currency=quote_currency,
            base_amount_near=base_amount,
            spot_rate=spot_rate,
            forward_rate=forward_rate,
            near_date=near_date,
            far_date=far_date,
        )

    def present_value(
        self,
        valuation_date: date,
        base_curve: Curve,
        quote_curve: Curve,
        current_spot: float,
    ) -> Tuple[float, float]:
        """
        Calculate present value of FX swap.

        Returns:
            (PV in base currency, PV in quote currency)
        """
        pv_near = self.near_leg.present_value(
            valuation_date, base_curve, quote_curve, current_spot
        )
        pv_far = self.far_leg.present_value(
            valuation_date, base_curve, quote_curve, current_spot
        )

        return (pv_near[0] + pv_far[0], pv_near[1] + pv_far[1])

    def implied_rate_differential(self) -> float:
        """
        Calculate implied interest rate differential.

        From swap points: r_quote - r_base ≈ swap_points / (spot * T)
        """
        days = (self.far_date - self.near_date).days
        t = days / 365.0
        if t <= 0:
            return 0.0

        return math.log(self.forward_rate / self.spot_rate) / t

    def roll_cost(self) -> float:
        """
        Calculate roll cost (carry).

        Roll cost = (Forward - Spot) * Notional
        """
        return (self.forward_rate - self.spot_rate) * self.base_amount_near

    def __repr__(self) -> str:
        return (f"FXSwap({self.pair}, "
                f"near={self.near_date} @ {self.spot_rate:.5f}, "
                f"far={self.far_date} @ {self.forward_rate:.5f}, "
                f"points={self.swap_points:.2f})")


def calculate_forward_points(
    spot_rate: float,
    base_curve: Curve,
    quote_curve: Curve,
    value_date: date,
    pip_factor: float = 10000,
) -> float:
    """
    Calculate forward points for a given value date.

    Args:
        spot_rate: Current spot rate
        base_curve: Base currency curve
        quote_curve: Quote currency curve
        value_date: Forward value date
        pip_factor: Pip conversion factor (10000 for most pairs)

    Returns:
        Forward points in pips
    """
    df_base = base_curve.discount_factor(value_date)
    df_quote = quote_curve.discount_factor(value_date)

    forward_rate = spot_rate * df_base / df_quote
    forward_points = (forward_rate - spot_rate) * pip_factor

    return forward_points


def forward_point_curve(
    spot_rate: float,
    base_curve: Curve,
    quote_curve: Curve,
    tenors: list,
    reference_date: date,
) -> dict:
    """
    Generate forward point curve for multiple tenors.

    Args:
        spot_rate: Spot rate
        base_curve: Base currency curve
        quote_curve: Quote currency curve
        tenors: List of tenors (e.g., ["1W", "1M", "3M", "6M", "1Y"])
        reference_date: Reference date

    Returns:
        Dictionary of {tenor: (value_date, forward_rate, forward_points)}
    """
    tenor_map = {
        "ON": 1, "TN": 2, "SN": 3, "SW": 7, "1W": 7, "2W": 14,
        "1M": 30, "2M": 60, "3M": 91, "6M": 182, "9M": 273, "1Y": 365,
        "2Y": 730, "3Y": 1095, "5Y": 1825,
    }

    results = {}

    for tenor in tenors:
        tenor_upper = tenor.upper()
        if tenor_upper in tenor_map:
            days = tenor_map[tenor_upper]
        else:
            # Try to parse custom tenor
            if tenor_upper.endswith("D"):
                days = int(tenor_upper[:-1])
            elif tenor_upper.endswith("W"):
                days = int(tenor_upper[:-1]) * 7
            elif tenor_upper.endswith("M"):
                days = int(tenor_upper[:-1]) * 30
            elif tenor_upper.endswith("Y"):
                days = int(tenor_upper[:-1]) * 365
            else:
                continue

        value_date = reference_date + timedelta(days=days)
        fwd_points = calculate_forward_points(
            spot_rate, base_curve, quote_curve, value_date
        )

        df_base = base_curve.discount_factor(value_date)
        df_quote = quote_curve.discount_factor(value_date)
        fwd_rate = spot_rate * df_base / df_quote

        results[tenor] = (value_date, fwd_rate, fwd_points)

    return results
