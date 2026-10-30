"""
Zero Coupon Swap (ZCS)

A zero coupon swap exchanges a single compounded payment at maturity
instead of periodic coupon payments.

Types:
- Fixed ZCS: Fixed rate compounded to maturity vs floating
- Float ZCS: Floating compounded to maturity vs fixed

Common uses:
- Liability matching
- Zero-coupon bond replication
- Duration management
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional, List, TYPE_CHECKING

from finceptquantlib.core.foundations.types import (
    Currency, Rate, as_tenor
)
from finceptquantlib.core.instruments.legs import (
    ZeroCouponLeg, FloatLeg
)

if TYPE_CHECKING:
    from finceptquantlib.curves.curve import Curve


# =============================================================================
# Zero Coupon Swap
# =============================================================================

@dataclass
class ZeroCouponSwap:
    """
    Zero Coupon Swap instrument.

    Exchanges a single compounded fixed payment at maturity against
    floating rate payments (or vice versa).

    In a pay-fixed ZCS:
    - Pay: Fixed rate compounded to maturity (single payment)
    - Receive: Floating rate (periodic or compounded)

    Args:
        effective_date: Start date
        termination_date: Maturity date
        notional: Notional amount
        fixed_rate: Fixed rate (annualized)
        currency: Currency
        pay_fixed: True = pay fixed, receive float
        fixed_compounding: Compounding frequency for fixed leg
        float_frequency: Payment frequency for float leg
        float_spread: Spread over float index
        float_index: Floating rate index name
        day_count_fixed: Day count for fixed leg
        day_count_float: Day count for float leg
    """
    effective_date: date
    termination_date: date
    notional: float
    fixed_rate: Rate
    currency: Currency
    pay_fixed: bool = True
    fixed_compounding: str = "1Y"  # Annual compounding
    float_frequency: str = "3M"
    float_spread: Rate = field(default_factory=lambda: Rate(0.0))
    float_index: str = ""
    day_count_fixed: str = "ACT/365"
    day_count_float: str = "ACT/360"

    _fixed_leg: Optional[ZeroCouponLeg] = field(default=None, repr=False)
    _float_leg: Optional[FloatLeg] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.fixed_rate, (float, int, str)):
            self.fixed_rate = Rate(self.fixed_rate)
        if isinstance(self.float_spread, (float, int, str)):
            self.float_spread = Rate(self.float_spread)
        if isinstance(self.currency, str):
            self.currency = Currency.from_string(self.currency)

        self._build_legs()

    def _build_legs(self) -> None:
        """Build the swap legs."""
        # Fixed zero coupon leg
        pay_receive_fixed = -1 if self.pay_fixed else 1
        self._fixed_leg = ZeroCouponLeg(
            effective_date=self.effective_date,
            termination_date=self.termination_date,
            notional_amount=self.notional,
            ccy=self.currency,
            rate=self.fixed_rate,
            day_count=self.day_count_fixed,
            compounding_frequency=self.fixed_compounding,
            pay_receive=pay_receive_fixed
        )

        # Floating leg (periodic payments)
        pay_receive_float = 1 if self.pay_fixed else -1
        self._float_leg = FloatLeg(
            effective_date=self.effective_date,
            termination_date=self.termination_date,
            notional_amount=self.notional,
            ccy=self.currency,
            spread=self.float_spread,
            index_name=self.float_index,
            frequency=self.float_frequency,
            day_count=self.day_count_float,
            pay_receive=pay_receive_float
        )

    @property
    def fixed_leg(self) -> ZeroCouponLeg:
        """Get the fixed zero coupon leg."""
        return self._fixed_leg

    @property
    def float_leg(self) -> FloatLeg:
        """Get the floating leg."""
        return self._float_leg

    @property
    def tenor(self) -> str:
        """Swap tenor as string."""
        days = (self.termination_date - self.effective_date).days
        years = days / 365.0
        if years >= 1:
            return f"{int(round(years))}Y"
        months = int(round(years * 12))
        return f"{months}M"

    def compounded_fixed_amount(self) -> float:
        """Calculate the compounded fixed payment at maturity."""
        tenor = as_tenor(self.fixed_compounding)
        dcf = self._year_fraction(
            self.effective_date,
            self.termination_date,
            self.day_count_fixed
        )

        # Number of compounding periods
        if tenor.total_months > 0:
            periods_per_year = 12 / tenor.total_months
        else:
            periods_per_year = 1

        n_periods = dcf * periods_per_year
        compounded_factor = (1 + self.fixed_rate.decimal / periods_per_year) ** n_periods - 1

        return self.notional * compounded_factor

    def npv(
        self,
        discount_curve: Curve,
        forecast_curve: Optional[Curve] = None,
        valuation_date: Optional[date] = None
    ) -> float:
        """
        Calculate NPV of the swap.

        Args:
            discount_curve: Curve for discounting
            forecast_curve: Curve for float rate projection
            valuation_date: Valuation date

        Returns:
            Net present value
        """
        fc = forecast_curve if forecast_curve is not None else discount_curve

        fixed_npv = self._fixed_leg.npv(discount_curve)
        float_npv = self._float_leg.npv(discount_curve, fc)

        return fixed_npv + float_npv

    def par_rate(
        self,
        discount_curve: Curve,
        forecast_curve: Optional[Curve] = None
    ) -> float:
        """
        Calculate par fixed rate.

        The rate that makes NPV = 0.

        Returns:
            Par rate as decimal
        """
        fc = forecast_curve if forecast_curve is not None else discount_curve

        # Float leg NPV
        float_npv = abs(self._float_leg.npv(discount_curve, fc))

        # Fixed leg annuity (PV of $1 at maturity)
        df = discount_curve.discount_factor(self.termination_date)
        dcf = self._year_fraction(
            self.effective_date,
            self.termination_date,
            self.day_count_fixed
        )

        # For ZCS with single payment: Par = Float_NPV / (Notional * DF * DCF)
        # This is approximate; exact solution requires solving for compounded rate
        if df * dcf * self.notional > 0:
            return float_npv / (df * dcf * self.notional)
        return 0.0

    def dv01(
        self,
        discount_curve: Curve,
        forecast_curve: Optional[Curve] = None
    ) -> float:
        """Calculate DV01 (dollar value of 1bp)."""
        fc = forecast_curve if forecast_curve is not None else discount_curve

        # Bump and reprice
        base_npv = self.npv(discount_curve, fc)

        # Create bumped swap
        bumped_rate = Rate(self.fixed_rate.decimal + 0.0001)
        bumped_swap = ZeroCouponSwap(
            effective_date=self.effective_date,
            termination_date=self.termination_date,
            notional=self.notional,
            fixed_rate=bumped_rate,
            currency=self.currency,
            pay_fixed=self.pay_fixed,
            fixed_compounding=self.fixed_compounding,
            float_frequency=self.float_frequency,
            float_spread=self.float_spread,
            float_index=self.float_index
        )
        bumped_npv = bumped_swap.npv(discount_curve, fc)

        return bumped_npv - base_npv

    def _year_fraction(self, start: date, end: date, day_count: str) -> float:
        """Calculate year fraction."""
        days = (end - start).days
        dc = day_count.upper().replace("/", "").replace(" ", "")
        if "360" in dc:
            return days / 360.0
        return days / 365.0

    def cashflows(
        self,
        forecast_curve: Optional[Curve] = None
    ) -> List[tuple[date, float, str]]:
        """
        Generate all cashflows.

        Returns list of (date, amount, description) tuples.
        """
        result = []

        # Fixed leg (single payment at maturity)
        fixed_cf = self.compounded_fixed_amount()
        if self.pay_fixed:
            fixed_cf = -fixed_cf
        result.append((self.termination_date, fixed_cf, "Fixed ZC payment"))

        # Float leg
        for cf_date, cf_amount in self._float_leg.cashflows(forecast_curve):
            result.append((cf_date, cf_amount, "Float payment"))

        return sorted(result, key=lambda x: x[0])

    def __repr__(self) -> str:
        direction = "Pay" if self.pay_fixed else "Receive"
        return (
            f"ZeroCouponSwap({direction} {self.fixed_rate} fixed, "
            f"{self.tenor}, notional={self.notional:,.0f} {self.currency.value})"
        )


# =============================================================================
# Zero-Zero Swap (Both legs zero coupon)
# =============================================================================

@dataclass
class ZeroZeroSwap:
    """
    Zero-Zero Swap: Both legs are zero coupon (single payments at maturity).

    Used for:
    - Pure rate differential trades
    - Basis trades between indices

    Args:
        effective_date: Start date
        termination_date: Maturity date
        notional: Notional amount
        fixed_rate: Fixed rate for pay leg
        currency: Currency
        float_index: Index for receive leg
        compounding_pay: Compounding frequency for pay leg
        compounding_receive: Compounding frequency for receive leg
    """
    effective_date: date
    termination_date: date
    notional: float
    fixed_rate: Rate
    currency: Currency
    compounding_pay: str = "1Y"
    compounding_receive: str = "1Y"
    day_count: str = "ACT/365"

    _pay_leg: Optional[ZeroCouponLeg] = field(default=None, repr=False)
    _receive_leg: Optional[ZeroCouponLeg] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.fixed_rate, (float, int, str)):
            self.fixed_rate = Rate(self.fixed_rate)
        if isinstance(self.currency, str):
            self.currency = Currency.from_string(self.currency)

        self._build_legs()

    def _build_legs(self) -> None:
        """Build both zero coupon legs."""
        self._pay_leg = ZeroCouponLeg(
            effective_date=self.effective_date,
            termination_date=self.termination_date,
            notional_amount=self.notional,
            ccy=self.currency,
            rate=self.fixed_rate,
            day_count=self.day_count,
            compounding_frequency=self.compounding_pay,
            pay_receive=-1
        )

        # Receive leg - floating (rate determined from curve)
        self._receive_leg = ZeroCouponLeg(
            effective_date=self.effective_date,
            termination_date=self.termination_date,
            notional_amount=self.notional,
            ccy=self.currency,
            rate=None,  # Floating
            day_count=self.day_count,
            compounding_frequency=self.compounding_receive,
            pay_receive=1
        )

    def npv(
        self,
        discount_curve: Curve,
        forecast_curve: Optional[Curve] = None
    ) -> float:
        """Calculate NPV."""
        fc = forecast_curve if forecast_curve is not None else discount_curve
        return (
            self._pay_leg.npv(discount_curve, fc) +
            self._receive_leg.npv(discount_curve, fc)
        )

    def __repr__(self) -> str:
        return (
            f"ZeroZeroSwap({self.fixed_rate}, "
            f"{self.effective_date} -> {self.termination_date})"
        )
