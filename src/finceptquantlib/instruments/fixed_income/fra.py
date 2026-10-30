"""
Forward Rate Agreements (FRAs)

Implements FRA pricing and analytics:
- Standard FRA pricing
- Settlement calculation
- Greeks and sensitivities

A FRA is an OTC contract to pay or receive a fixed rate vs. a floating rate
for a future period. Settlement is typically at the start of the period.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Union, Optional

from finceptquantlib.curves.curve import Curve
from finceptquantlib.scheduling.calendar import Calendar
from finceptquantlib.scheduling.daycount import DayCountConvention, _get_convention
from finceptquantlib.scheduling.adjustment import BusinessDayAdjustment


@dataclass
class FRAResult:
    """Result of FRA valuation."""
    present_value: float
    forward_rate: float
    settlement_amount: float
    dv01: float
    settlement_date: date
    start_date: date
    end_date: date


class ForwardRateAgreement:
    """
    Forward Rate Agreement (FRA).

    A FRA is an agreement to exchange a fixed rate for a floating rate
    over a specified future period. The payoff is:

    Payer FRA: Notional * (R_market - R_fixed) * dcf / (1 + R_market * dcf)
    Receiver FRA: Notional * (R_fixed - R_market) * dcf / (1 + R_market * dcf)

    where settlement is at the start of the period (discounted at market rate).

    FRA Convention: "3x6" means FRA starting in 3 months, ending in 6 months.
    """

    def __init__(
        self,
        notional: float,
        start_date: date,
        end_date: date,
        fixed_rate: float,
        is_payer: bool = True,
        day_count: Union[str, DayCountConvention] = "ACT/360",
        calendar: Optional[Calendar] = None,
        fixing_days: int = 2,
        settlement_type: str = "in_advance",
    ):
        """
        Initialize FRA.

        Args:
            notional: Notional principal
            start_date: FRA period start date
            end_date: FRA period end date
            fixed_rate: Fixed FRA rate
            is_payer: True = pay fixed/receive floating
            day_count: Day count convention
            calendar: Business day calendar
            fixing_days: Days before start for rate fixing
            settlement_type: "in_advance" (standard) or "in_arrears"
        """
        self.notional = notional
        self.start_date = start_date
        self.end_date = end_date
        self.fixed_rate = fixed_rate
        self.is_payer = is_payer

        if isinstance(day_count, str):
            day_count = _get_convention(day_count)
        self.day_count = day_count

        self.calendar = calendar
        self.fixing_days = fixing_days
        self.settlement_type = settlement_type

    @classmethod
    def from_convention(
        cls,
        notional: float,
        trade_date: date,
        start_months: int,
        end_months: int,
        fixed_rate: float,
        is_payer: bool = True,
        day_count: Union[str, DayCountConvention] = "ACT/360",
        calendar: Optional[Calendar] = None,
    ) -> "ForwardRateAgreement":
        """
        Create FRA from market convention (e.g., 3x6, 6x12).

        Args:
            notional: Notional principal
            trade_date: Trade date
            start_months: Months to FRA start
            end_months: Months to FRA end
            fixed_rate: Fixed FRA rate
            is_payer: True = pay fixed/receive floating
            day_count: Day count convention
            calendar: Business day calendar

        Returns:
            ForwardRateAgreement

        Example:
            fra = ForwardRateAgreement.from_convention(
                notional=10_000_000,
                trade_date=date(2024, 1, 15),
                start_months=3,
                end_months=6,
                fixed_rate=0.05,  # 5%
            )
        """
        # Calculate start and end dates
        start_date = trade_date + timedelta(days=start_months * 30)  # Approximate
        end_date = trade_date + timedelta(days=end_months * 30)

        # Adjust for business days if calendar provided
        if calendar:
            adj = BusinessDayAdjustment("modified_following", calendar)
            start_date = adj.adjust(start_date)
            end_date = adj.adjust(end_date)

        return cls(
            notional=notional,
            start_date=start_date,
            end_date=end_date,
            fixed_rate=fixed_rate,
            is_payer=is_payer,
            day_count=day_count,
            calendar=calendar,
        )

    @property
    def fixing_date(self) -> date:
        """Date when the floating rate is fixed."""
        fixing = self.start_date - timedelta(days=self.fixing_days)
        if self.calendar:
            adj = BusinessDayAdjustment("preceding", self.calendar)
            fixing = adj.adjust(fixing)
        return fixing

    @property
    def settlement_date(self) -> date:
        """Date when settlement occurs."""
        if self.settlement_type == "in_advance":
            return self.start_date
        else:
            return self.end_date

    @property
    def day_count_fraction(self) -> float:
        """Day count fraction for the FRA period."""
        return self.day_count.year_fraction(self.start_date, self.end_date)

    def forward_rate(self, curve: Curve) -> float:
        """
        Calculate the implied forward rate from the curve.

        F = (DF(t1)/DF(t2) - 1) / dcf
        """
        t1 = curve.time_to_maturity(self.start_date)
        t2 = curve.time_to_maturity(self.end_date)

        if t1 <= 0:
            # Start date is in the past
            return curve.forward_rate(0.001, max(t2, 0.002))

        df1 = curve.discount_factor(self.start_date)
        df2 = curve.discount_factor(self.end_date)
        dcf = self.day_count_fraction

        return (df1 / df2 - 1) / dcf

    def settlement_amount(
        self,
        valuation_date: date,
        market_rate: float,
    ) -> float:
        """
        Calculate FRA settlement amount.

        Settlement = N * (R_market - R_fixed) * dcf / (1 + R_market * dcf)
        for payer FRA (receive floating).
        """
        dcf = self.day_count_fraction
        rate_diff = market_rate - self.fixed_rate

        if not self.is_payer:
            rate_diff = -rate_diff

        # Discounted to settlement date
        settlement = self.notional * rate_diff * dcf / (1 + market_rate * dcf)

        return settlement

    def present_value(
        self,
        valuation_date: date,
        curve: Curve,
        market_rate: Optional[float] = None,
    ) -> float:
        """
        Calculate present value of FRA.

        Args:
            valuation_date: Valuation date
            curve: Discount/projection curve
            market_rate: Fixed market rate (if already known)

        Returns:
            Present value
        """
        if market_rate is None:
            market_rate = self.forward_rate(curve)

        settlement = self.settlement_amount(valuation_date, market_rate)

        # Discount from settlement date to valuation date
        df = curve.discount_factor(self.settlement_date)

        return settlement * df

    def dv01(
        self,
        valuation_date: date,
        curve: Curve,
        bump_size: float = 1.0,
    ) -> float:
        """
        Calculate DV01 (dollar value of 1bp move in rates).

        Args:
            valuation_date: Valuation date
            curve: Discount curve
            bump_size: Bump size in basis points

        Returns:
            DV01
        """
        base_pv = self.present_value(valuation_date, curve)
        bumped_curve = curve.shift(bump_size)
        bumped_pv = self.present_value(valuation_date, bumped_curve)

        return bumped_pv - base_pv

    def gamma(
        self,
        valuation_date: date,
        curve: Curve,
        bump_size: float = 1.0,
    ) -> float:
        """Calculate gamma (convexity)."""
        base_pv = self.present_value(valuation_date, curve)
        up_curve = curve.shift(bump_size)
        down_curve = curve.shift(-bump_size)

        pv_up = self.present_value(valuation_date, up_curve)
        pv_down = self.present_value(valuation_date, down_curve)

        return (pv_up - 2 * base_pv + pv_down) / (bump_size ** 2)

    def value(
        self,
        valuation_date: date,
        curve: Curve,
    ) -> FRAResult:
        """
        Full valuation with all analytics.

        Returns:
            FRAResult with all metrics
        """
        fwd_rate = self.forward_rate(curve)
        pv = self.present_value(valuation_date, curve, fwd_rate)
        settlement = self.settlement_amount(valuation_date, fwd_rate)
        dv01 = self.dv01(valuation_date, curve)

        return FRAResult(
            present_value=pv,
            forward_rate=fwd_rate,
            settlement_amount=settlement,
            dv01=dv01,
            settlement_date=self.settlement_date,
            start_date=self.start_date,
            end_date=self.end_date,
        )

    def break_even_rate(
        self,
        valuation_date: date,
        curve: Curve,
    ) -> float:
        """
        Calculate break-even rate.

        The market rate at which the FRA has zero value.
        For a new FRA, this equals the forward rate.
        """
        return self.forward_rate(curve)

    def __repr__(self) -> str:
        direction = "Payer" if self.is_payer else "Receiver"
        return (f"FRA({direction}, notional={self.notional:,.0f}, "
                f"rate={self.fixed_rate:.4%}, "
                f"{self.start_date} to {self.end_date})")


class FRAStrip:
    """
    Strip of consecutive FRAs.

    Useful for hedging or creating synthetic swap exposures.
    """

    def __init__(
        self,
        notional: float,
        start_date: date,
        num_periods: int,
        tenor_months: int,
        fixed_rates: Union[float, list],
        is_payer: bool = True,
        day_count: Union[str, DayCountConvention] = "ACT/360",
        calendar: Optional[Calendar] = None,
    ):
        """
        Initialize FRA strip.

        Args:
            notional: Notional per FRA
            start_date: First FRA start date
            num_periods: Number of FRAs in strip
            tenor_months: Months per FRA period
            fixed_rates: Fixed rate(s) - single value or list
            is_payer: Direction for all FRAs
            day_count: Day count convention
            calendar: Business day calendar
        """
        self.notional = notional
        self.fras: list[ForwardRateAgreement] = []

        if isinstance(fixed_rates, (int, float)):
            fixed_rates = [fixed_rates] * num_periods

        current_date = start_date
        for i, rate in enumerate(fixed_rates):
            end_date = current_date + timedelta(days=tenor_months * 30)
            if calendar:
                adj = BusinessDayAdjustment("modified_following", calendar)
                end_date = adj.adjust(end_date)

            fra = ForwardRateAgreement(
                notional=notional,
                start_date=current_date,
                end_date=end_date,
                fixed_rate=rate,
                is_payer=is_payer,
                day_count=day_count,
                calendar=calendar,
            )
            self.fras.append(fra)
            current_date = end_date

    def present_value(
        self,
        valuation_date: date,
        curve: Curve,
    ) -> float:
        """Calculate total present value of strip."""
        return sum(fra.present_value(valuation_date, curve) for fra in self.fras)

    def dv01(
        self,
        valuation_date: date,
        curve: Curve,
    ) -> float:
        """Calculate total DV01 of strip."""
        return sum(fra.dv01(valuation_date, curve) for fra in self.fras)

    def par_rates(self, curve: Curve) -> list[float]:
        """Get forward rates for each FRA."""
        return [fra.forward_rate(curve) for fra in self.fras]

    def __len__(self) -> int:
        return len(self.fras)

    def __iter__(self):
        return iter(self.fras)
