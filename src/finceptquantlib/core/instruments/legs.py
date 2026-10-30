"""
Leg Module - Collections of Periods

Provides leg types that represent sequences of payments:
- FixedLeg: Sequence of fixed coupon periods
- FloatLeg: Sequence of floating coupon periods
- CompoundingLeg: RFR compounded leg
- ZeroCouponLeg: Single payment at maturity
- InflationLeg: Inflation-indexed payments
- CustomLeg: User-defined cashflows
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional, List, Dict, Iterator, Union, TYPE_CHECKING

from finceptquantlib.core.foundations.types import (
    Rate, Currency, Tenor, NotionalSchedule, StubType, as_tenor
)
from finceptquantlib.core.instruments.periods import (
    BasePeriod, FixedCouponPeriod, FloatCouponPeriod,
    CompoundedRFRPeriod, InflationPeriod, PrincipalExchange,
    FeePayment, PeriodList, CompoundingMethod,
    FixingType, ObservationMethod, ExchangeType
)

if TYPE_CHECKING:
    from finceptquantlib.curves.curve import Curve


# =============================================================================
# Shared Schedule Generation
# =============================================================================

def generate_schedule_dates(
    effective_date: date,
    termination_date: date,
    frequency: str,
    stub: StubType = StubType.SHORT_FRONT,
    business_day_adjust: bool = False,
) -> List[date]:
    """
    Generate schedule dates with stub type support.

    Args:
        effective_date: Start date
        termination_date: End date
        frequency: Tenor string (e.g., "3M", "6M", "1Y")
        stub: Stub period type
        business_day_adjust: If True, adjust dates to next business day (Mon-Fri)

    Returns:
        List of schedule dates from effective to termination
    """
    tenor = as_tenor(frequency)

    if stub in (StubType.SHORT_FRONT, StubType.LONG_FRONT):
        # Generate backwards from termination date
        dates = [termination_date]
        current = termination_date
        while True:
            # Subtract tenor
            prev_date = _subtract_tenor(current, tenor)
            if prev_date <= effective_date:
                if stub == StubType.LONG_FRONT and len(dates) > 1:
                    # Merge first two periods for long stub
                    pass  # effective_date will be the start
                dates.insert(0, effective_date)
                break
            dates.insert(0, prev_date)
            current = prev_date
    elif stub in (StubType.SHORT_BACK, StubType.LONG_BACK):
        # Generate forwards from effective date
        dates = [effective_date]
        current = effective_date
        while True:
            next_date = tenor.add_to_date(current)
            if next_date >= termination_date:
                if stub == StubType.LONG_BACK and len(dates) > 1:
                    # Merge last two periods for long stub
                    pass  # termination_date will be the end
                if dates[-1] != termination_date:
                    dates.append(termination_date)
                break
            dates.append(next_date)
            current = next_date
    else:
        # StubType.NONE or default — forward generation
        dates = [effective_date]
        current = effective_date
        while True:
            next_date = tenor.add_to_date(current)
            if next_date >= termination_date:
                if dates[-1] != termination_date:
                    dates.append(termination_date)
                break
            dates.append(next_date)
            current = next_date

    if business_day_adjust:
        dates = [_adjust_business_day(d) for d in dates]

    return dates


def _subtract_tenor(dt: date, tenor: Tenor) -> date:
    """Subtract a tenor from a date."""
    year = dt.year - tenor.years
    month = dt.month - tenor.months
    while month < 1:
        month += 12
        year -= 1
    day = min(dt.day, _days_in_month(year, month))
    result = date(year, month, day)
    result -= timedelta(weeks=tenor.weeks, days=tenor.days)
    return result


def _days_in_month(year: int, month: int) -> int:
    """Get number of days in a month."""
    if month in (1, 3, 5, 7, 8, 10, 12):
        return 31
    elif month in (4, 6, 9, 11):
        return 30
    else:  # February
        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            return 29
        return 28


def _adjust_business_day(dt: date) -> date:
    """Adjust to next business day (Modified Following convention)."""
    # Saturday -> Monday, Sunday -> Monday
    if dt.weekday() == 5:  # Saturday
        adjusted = dt + timedelta(days=2)
    elif dt.weekday() == 6:  # Sunday
        adjusted = dt + timedelta(days=1)
    else:
        return dt

    # Modified Following: if month rolls, go backwards
    if adjusted.month != dt.month:
        # Go to previous Friday instead
        adjusted = dt
        while adjusted.weekday() >= 5:
            adjusted -= timedelta(days=1)
    return adjusted


# =============================================================================
# Base Leg Protocol
# =============================================================================

class BaseLeg(ABC):
    """
    Abstract base class for all leg types.

    A leg is a sequence of periods representing one side of a swap
    or other multi-period instrument.
    """

    @property
    @abstractmethod
    def currency(self) -> Currency:
        """Currency of the leg."""

    @property
    @abstractmethod
    def start_date(self) -> date:
        """Start date of the leg."""

    @property
    @abstractmethod
    def end_date(self) -> date:
        """End date (maturity) of the leg."""

    @property
    @abstractmethod
    def notional(self) -> float:
        """Initial notional amount."""

    @property
    @abstractmethod
    def periods(self) -> PeriodList:
        """List of periods in the leg."""

    @abstractmethod
    def cashflows(self, curve: Optional[Curve] = None) -> List[tuple[date, float]]:
        """
        Generate list of (date, amount) cashflows.

        Args:
            curve: Optional curve for floating rate projection

        Returns:
            List of (payment_date, cashflow_amount) tuples
        """

    @abstractmethod
    def npv(
        self,
        discount_curve: Curve,
        forecast_curve: Optional[Curve] = None,
        valuation_date: Optional[date] = None
    ) -> float:
        """Calculate NPV of the leg."""

    def __iter__(self) -> Iterator[BasePeriod]:
        """Iterate over periods."""
        return iter(self.periods)

    def __len__(self) -> int:
        """Number of periods."""
        return len(self.periods)


# =============================================================================
# Fixed Leg
# =============================================================================

@dataclass
class CoreFixedLeg(BaseLeg):
    """
    Fixed rate payment leg (core implementation).

    Generates a sequence of fixed coupon payments based on schedule.

    Args:
        effective_date: Start date of the leg
        termination_date: End date of the leg
        coupon_rate: Fixed coupon rate
        notional_amount: Notional principal (or schedule)
        ccy: Currency
        frequency: Payment frequency ("1M", "3M", "6M", "1Y")
        day_count: Day count convention
        payment_delay: Days after accrual end for payment
        stub: Stub period type
        pay_receive: 1 = receive, -1 = pay
    """
    effective_date: date
    termination_date: date
    coupon_rate: Rate
    notional_amount: Union[float, NotionalSchedule]
    ccy: Currency
    frequency: str = "6M"
    day_count: str = "ACT/360"
    payment_delay: int = 0
    stub: StubType = StubType.SHORT_FRONT
    pay_receive: int = 1
    _periods: PeriodList = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.coupon_rate, (float, int, str)):
            self.coupon_rate = Rate(self.coupon_rate)
        if isinstance(self.ccy, str):
            self.ccy = Currency.from_string(self.ccy)
        if isinstance(self.notional_amount, (int, float)):
            self.notional_amount = NotionalSchedule.constant(float(self.notional_amount))

        self._build_periods()

    def _build_periods(self) -> None:
        """Build the period list from schedule."""
        self._periods = []
        schedule_dates = generate_schedule_dates(
            self.effective_date, self.termination_date,
            self.frequency, self.stub
        )

        for i in range(len(schedule_dates) - 1):
            start = schedule_dates[i]
            end = schedule_dates[i + 1]
            payment = end + timedelta(days=self.payment_delay)
            notional = self.notional_amount.get_notional(i)

            period = FixedCouponPeriod(
                start=start,
                end=end,
                payment=payment,
                rate=self.coupon_rate,
                notional_amount=notional * self.pay_receive,
                ccy=self.ccy,
                day_count=self.day_count
            )
            self._periods.append(period)

    @property
    def currency(self) -> Currency:
        return self.ccy

    @property
    def start_date(self) -> date:
        return self.effective_date

    @property
    def end_date(self) -> date:
        return self.termination_date

    @property
    def notional(self) -> float:
        return self.notional_amount.get_notional(0)

    @property
    def periods(self) -> PeriodList:
        return self._periods

    def cashflows(self, curve: Optional[Curve] = None) -> List[tuple[date, float]]:
        return [(p.payment_date, p.cashflow(curve)) for p in self._periods]

    def npv(
        self,
        discount_curve: Curve,
        forecast_curve: Optional[Curve] = None,
        valuation_date: Optional[date] = None
    ) -> float:
        return sum(p.npv(discount_curve, forecast_curve) for p in self._periods)

    def dv01(self, discount_curve: Curve) -> float:
        """Calculate DV01 (dollar value of 1bp) using automatic differentiation."""
        from finceptquantlib.core.foundations.autodiff import Dual

        # Use Dual number for the coupon rate to get exact derivative
        rate_dual = Dual(self.coupon_rate.decimal, 1.0)

        # Compute NPV as function of rate
        npv_val = 0.0
        for p in self._periods:
            dcf = p.accrual_fraction()
            cf = p.notional_amount * rate_dual * dcf
            df = discount_curve.discount_factor(p.payment_date)
            npv_val += cf * df

        # DV01 = dNPV/dRate * 0.0001
        if isinstance(npv_val, Dual):
            return npv_val.deriv * 0.0001
        return 0.0

    def annuity(self, discount_curve: Curve) -> float:
        """Calculate annuity (PV of 1% coupon)."""
        return sum(
            p.accrual_fraction() * discount_curve.discount_factor(p.payment_date)
            * abs(p.notional)
            for p in self._periods
        ) / 100.0

    def __repr__(self) -> str:
        direction = "receive" if self.pay_receive > 0 else "pay"
        return (
            f"CoreFixedLeg({direction} {self.coupon_rate}, "
            f"{self.effective_date} -> {self.termination_date}, "
            f"notional={self.notional:,.0f} {self.ccy.value})"
        )


# =============================================================================
# Float Leg
# =============================================================================

@dataclass
class FloatLeg(BaseLeg):
    """
    Floating rate payment leg.

    Args:
        effective_date: Start date of the leg
        termination_date: End date of the leg
        spread: Spread over index rate
        notional_amount: Notional principal (or schedule)
        ccy: Currency
        index_name: Name of reference index (e.g., "SOFR", "EURIBOR")
        frequency: Payment frequency
        day_count: Day count convention
        fixing_lag: Days before period start for fixing
        payment_delay: Days after accrual end for payment
        pay_receive: 1 = receive, -1 = pay
    """
    effective_date: date
    termination_date: date
    notional_amount: Union[float, NotionalSchedule]
    ccy: Currency
    spread: Rate = field(default_factory=lambda: Rate(0.0))
    index_name: str = ""
    frequency: str = "3M"
    day_count: str = "ACT/360"
    fixing_lag: int = 2
    payment_delay: int = 0
    fixing_type: FixingType = FixingType.IN_ADVANCE
    pay_receive: int = 1
    _periods: List[FloatCouponPeriod] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.spread, (float, int, str)):
            self.spread = Rate(self.spread)
        if isinstance(self.ccy, str):
            self.ccy = Currency.from_string(self.ccy)
        if isinstance(self.notional_amount, (int, float)):
            self.notional_amount = NotionalSchedule.constant(float(self.notional_amount))

        self._build_periods()

    def _build_periods(self) -> None:
        """Build the period list from schedule."""
        self._periods = []
        schedule_dates = generate_schedule_dates(
            self.effective_date, self.termination_date, self.frequency
        )

        for i in range(len(schedule_dates) - 1):
            start = schedule_dates[i]
            end = schedule_dates[i + 1]
            fixing = start - timedelta(days=self.fixing_lag)
            payment = end + timedelta(days=self.payment_delay)
            notional = self.notional_amount.get_notional(i)

            period = FloatCouponPeriod(
                start=start,
                end=end,
                payment=payment,
                fixing_date=fixing,
                notional_amount=notional * self.pay_receive,
                ccy=self.ccy,
                spread=self.spread,
                day_count=self.day_count,
                fixing_type=self.fixing_type,
                index_name=self.index_name
            )
            self._periods.append(period)

    @property
    def currency(self) -> Currency:
        return self.ccy

    @property
    def start_date(self) -> date:
        return self.effective_date

    @property
    def end_date(self) -> date:
        return self.termination_date

    @property
    def notional(self) -> float:
        return self.notional_amount.get_notional(0)

    @property
    def periods(self) -> PeriodList:
        return self._periods

    def set_fixings(self, fixings: Dict[date, float]) -> None:
        """Set fixing rates for periods."""
        for period in self._periods:
            if period.fixing_date in fixings:
                period.set_fixing(fixings[period.fixing_date])

    def cashflows(self, curve: Optional[Curve] = None) -> List[tuple[date, float]]:
        return [(p.payment_date, p.cashflow(curve)) for p in self._periods]

    def npv(
        self,
        discount_curve: Curve,
        forecast_curve: Optional[Curve] = None,
        valuation_date: Optional[date] = None
    ) -> float:
        fc = forecast_curve if forecast_curve is not None else discount_curve
        return sum(p.npv(discount_curve, fc) for p in self._periods)

    def spread_dv01(self, discount_curve: Curve, forecast_curve: Optional[Curve] = None) -> float:
        """Calculate spread DV01 using automatic differentiation."""
        from finceptquantlib.core.foundations.autodiff import Dual

        fc = forecast_curve if forecast_curve is not None else discount_curve
        spread_dual = Dual(self.spread.decimal, 1.0)

        npv_val = 0.0
        for p in self._periods:
            # Get base rate from fixing or curve
            if p._fixed_rate is not None:
                base_rate = p._fixed_rate
            else:
                base_rate = fc.forward_rate(p.start, p.end)

            dcf = p.accrual_fraction()
            total_rate = base_rate + spread_dual
            cf = p.notional_amount * total_rate * dcf
            df = discount_curve.discount_factor(p.payment_date)
            npv_val += cf * df

        if isinstance(npv_val, Dual):
            return npv_val.deriv * 0.0001
        return 0.0

    def __repr__(self) -> str:
        direction = "receive" if self.pay_receive > 0 else "pay"
        spread_str = f"+{self.spread}" if self.spread.decimal > 0 else str(self.spread)
        return (
            f"FloatLeg({direction} {self.index_name}{spread_str}, "
            f"{self.effective_date} -> {self.termination_date}, "
            f"notional={self.notional:,.0f} {self.ccy.value})"
        )


# =============================================================================
# Compounding Leg (RFR)
# =============================================================================

@dataclass
class CompoundingLeg(BaseLeg):
    """
    RFR compounded leg (SOFR, SONIA, ESTR, etc.).

    Implements ISDA RFR compounding conventions.

    Args:
        effective_date: Start date
        termination_date: End date
        notional_amount: Notional principal
        ccy: Currency
        spread: Spread over compounded rate
        index_name: RFR index name
        frequency: Payment frequency
        day_count: Day count convention
        observation_method: ISDA observation method
        lookback_days: Days for lookback/lockout
        payment_delay: Days payment is delayed
    """
    effective_date: date
    termination_date: date
    notional_amount: Union[float, NotionalSchedule]
    ccy: Currency
    spread: Rate = field(default_factory=lambda: Rate(0.0))
    index_name: str = "SOFR"
    frequency: str = "3M"
    day_count: str = "ACT/360"
    observation_method: ObservationMethod = ObservationMethod.PAYMENT_DELAY
    lookback_days: int = 2
    payment_delay: int = 2
    compounding_method: CompoundingMethod = CompoundingMethod.SPREAD_EXCLUSIVE
    pay_receive: int = 1
    _periods: List[CompoundedRFRPeriod] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.spread, (float, int, str)):
            self.spread = Rate(self.spread)
        if isinstance(self.ccy, str):
            self.ccy = Currency.from_string(self.ccy)
        if isinstance(self.notional_amount, (int, float)):
            self.notional_amount = NotionalSchedule.constant(float(self.notional_amount))

        self._build_periods()

    def _build_periods(self) -> None:
        """Build compounded RFR periods."""
        self._periods = []
        schedule_dates = generate_schedule_dates(
            self.effective_date, self.termination_date, self.frequency
        )

        for i in range(len(schedule_dates) - 1):
            start = schedule_dates[i]
            end = schedule_dates[i + 1]
            payment = end + timedelta(days=self.payment_delay)
            notional = self.notional_amount.get_notional(i)

            period = CompoundedRFRPeriod(
                start=start,
                end=end,
                payment=payment,
                notional_amount=notional * self.pay_receive,
                ccy=self.ccy,
                spread=self.spread,
                day_count=self.day_count,
                observation_method=self.observation_method,
                lookback_days=self.lookback_days,
                payment_delay_days=self.payment_delay,
                compounding_method=self.compounding_method
            )
            self._periods.append(period)

    @property
    def currency(self) -> Currency:
        return self.ccy

    @property
    def start_date(self) -> date:
        return self.effective_date

    @property
    def end_date(self) -> date:
        return self.termination_date

    @property
    def notional(self) -> float:
        return self.notional_amount.get_notional(0)

    @property
    def periods(self) -> PeriodList:
        return self._periods

    def set_daily_rates(self, rates: Dict[date, float]) -> None:
        """Set daily fixing rates for all periods."""
        for period in self._periods:
            period.set_daily_rates(rates)

    def cashflows(self, curve: Optional[Curve] = None) -> List[tuple[date, float]]:
        return [(p.payment_date, p.cashflow(curve)) for p in self._periods]

    def npv(
        self,
        discount_curve: Curve,
        forecast_curve: Optional[Curve] = None,
        valuation_date: Optional[date] = None
    ) -> float:
        fc = forecast_curve if forecast_curve is not None else discount_curve
        return sum(p.npv(discount_curve, fc) for p in self._periods)

    def __repr__(self) -> str:
        direction = "receive" if self.pay_receive > 0 else "pay"
        return (
            f"CompoundingLeg({direction} {self.index_name}+{self.spread}, "
            f"{self.observation_method.name}, "
            f"{self.effective_date} -> {self.termination_date})"
        )


# =============================================================================
# Zero Coupon Leg
# =============================================================================

@dataclass
class ZeroCouponLeg(BaseLeg):
    """
    Zero coupon leg with single payment at maturity.

    Used for:
    - Zero coupon swaps
    - Zero coupon inflation swaps
    - Bullet loans

    Args:
        effective_date: Start date
        termination_date: End date (payment date)
        rate: Fixed rate (for fixed ZC leg) or None for floating
        notional_amount: Notional principal
        ccy: Currency
        day_count: Day count convention for accrual
        compounding: Compounding frequency for rate
        pay_receive: 1 = receive, -1 = pay
    """
    effective_date: date
    termination_date: date
    notional_amount: float
    ccy: Currency
    rate: Optional[Rate] = None
    day_count: str = "ACT/365"
    compounding_frequency: str = "1Y"
    pay_receive: int = 1
    _period: Optional[BasePeriod] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.rate, (float, int, str)):
            self.rate = Rate(self.rate)
        if isinstance(self.ccy, str):
            self.ccy = Currency.from_string(self.ccy)
        self._build_period()

    def _build_period(self) -> None:
        """Build the single zero coupon period."""
        if self.rate is not None:
            # Fixed zero coupon
            self._period = FixedCouponPeriod(
                start=self.effective_date,
                end=self.termination_date,
                payment=self.termination_date,
                rate=self.rate,
                notional_amount=self.notional_amount * self.pay_receive,
                ccy=self.ccy,
                day_count=self.day_count
            )
        else:
            # Floating zero coupon - will need curve for pricing
            self._period = FloatCouponPeriod(
                start=self.effective_date,
                end=self.termination_date,
                payment=self.termination_date,
                fixing_date=self.effective_date,
                notional_amount=self.notional_amount * self.pay_receive,
                ccy=self.ccy,
                day_count=self.day_count
            )

    @property
    def currency(self) -> Currency:
        return self.ccy

    @property
    def start_date(self) -> date:
        return self.effective_date

    @property
    def end_date(self) -> date:
        return self.termination_date

    @property
    def notional(self) -> float:
        return self.notional_amount

    @property
    def periods(self) -> PeriodList:
        return [self._period] if self._period else []

    def compounded_rate(self, curve: Optional[Curve] = None) -> float:
        """Calculate the effective compounded rate."""
        if self.rate is not None:
            # Compound fixed rate over the period
            dcf = self._period.accrual_fraction()
            tenor = as_tenor(self.compounding_frequency)
            periods = dcf * 12 / tenor.total_months if tenor.total_months > 0 else dcf

            growth = 1 + self.rate.decimal / (12 / tenor.total_months)
            if growth <= 0:
                # A non-positive growth factor raised to a fractional period count is complex.
                raise ValueError(
                    f"Zero-coupon rate {self.rate.decimal} loses 100% or more per compounding "
                    "period; the compounded rate is undefined"
                )
            return growth ** periods - 1
        elif curve is not None:
            return curve.forward_rate(self.effective_date, self.termination_date)
        return 0.0

    def cashflow_amount(self, curve: Optional[Curve] = None) -> float:
        """Calculate the single cashflow amount."""
        compounded = self.compounded_rate(curve)
        return self.notional_amount * compounded * self.pay_receive

    def cashflows(self, curve: Optional[Curve] = None) -> List[tuple[date, float]]:
        return [(self.termination_date, self.cashflow_amount(curve))]

    def npv(
        self,
        discount_curve: Curve,
        forecast_curve: Optional[Curve] = None,
        valuation_date: Optional[date] = None
    ) -> float:
        fc = forecast_curve if forecast_curve is not None else discount_curve
        cf = self.cashflow_amount(fc)
        df = discount_curve.discount_factor(self.termination_date)
        return cf * df

    def __repr__(self) -> str:
        direction = "receive" if self.pay_receive > 0 else "pay"
        rate_str = str(self.rate) if self.rate else "floating"
        return (
            f"ZeroCouponLeg({direction} {rate_str}, "
            f"{self.effective_date} -> {self.termination_date})"
        )


# =============================================================================
# Inflation Leg
# =============================================================================

@dataclass
class InflationLeg(BaseLeg):
    """
    Inflation-indexed payment leg.

    Args:
        effective_date: Start date
        termination_date: End date
        base_index: Base inflation index value
        notional_amount: Notional principal
        ccy: Currency
        index_name: Inflation index name (e.g., "CPI-U", "HICP")
        frequency: Payment frequency
        day_count: Day count convention
        index_lag_months: Months lag for index lookup
        pay_receive: 1 = receive, -1 = pay
    """
    effective_date: date
    termination_date: date
    base_index: float
    notional_amount: float
    ccy: Currency
    index_name: str = "CPI"
    frequency: str = "1Y"
    day_count: str = "ACT/ACT"
    index_lag_months: int = 3
    interpolation: bool = True
    pay_receive: int = 1
    _periods: List[InflationPeriod] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.ccy, str):
            self.ccy = Currency.from_string(self.ccy)
        self._build_periods()

    def _build_periods(self) -> None:
        """Build inflation periods."""
        self._periods = []
        schedule_dates = generate_schedule_dates(
            self.effective_date, self.termination_date, self.frequency
        )

        for i in range(len(schedule_dates) - 1):
            start = schedule_dates[i]
            end = schedule_dates[i + 1]

            period = InflationPeriod(
                start=start,
                end=end,
                payment=end,
                notional_amount=self.notional_amount * self.pay_receive,
                ccy=self.ccy,
                base_index=self.base_index,
                index_lag_months=self.index_lag_months,
                interpolation=self.interpolation,
                day_count=self.day_count
            )
            self._periods.append(period)

    @property
    def currency(self) -> Currency:
        return self.ccy

    @property
    def start_date(self) -> date:
        return self.effective_date

    @property
    def end_date(self) -> date:
        return self.termination_date

    @property
    def notional(self) -> float:
        return self.notional_amount

    @property
    def periods(self) -> PeriodList:
        return self._periods

    def set_index_values(self, index_values: Dict[date, float]) -> None:
        """Set inflation index values for periods."""
        for period in self._periods:
            ref_date = period.index_reference_date(period.end)
            if ref_date in index_values:
                period.set_end_index(index_values[ref_date])

    def cashflows(self, curve: Optional[Curve] = None) -> List[tuple[date, float]]:
        return [(p.payment_date, p.cashflow(curve)) for p in self._periods]

    def npv(
        self,
        discount_curve: Curve,
        forecast_curve: Optional[Curve] = None,
        valuation_date: Optional[date] = None
    ) -> float:
        return sum(p.npv(discount_curve, forecast_curve) for p in self._periods)

    def __repr__(self) -> str:
        direction = "receive" if self.pay_receive > 0 else "pay"
        return (
            f"InflationLeg({direction} {self.index_name}, "
            f"base={self.base_index:.2f}, "
            f"{self.effective_date} -> {self.termination_date})"
        )


# =============================================================================
# Custom Leg
# =============================================================================

@dataclass
class CustomLeg(BaseLeg):
    """
    User-defined cashflow leg.

    Allows arbitrary cashflow sequences for non-standard instruments.

    Args:
        cashflow_schedule: List of (date, amount) tuples
        ccy: Currency
        description: Description of the leg
    """
    cashflow_schedule: List[tuple[date, float]]
    ccy: Currency
    description: str = "Custom"
    _periods: PeriodList = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.ccy, str):
            self.ccy = Currency.from_string(self.ccy)
        self._build_periods()

    def _build_periods(self) -> None:
        """Build fee payment periods from cashflows."""
        self._periods = []
        for payment_date, amount in self.cashflow_schedule:
            period = FeePayment(
                fee_date=payment_date,
                amount=amount,
                ccy=self.ccy,
                description=self.description
            )
            self._periods.append(period)

    @property
    def currency(self) -> Currency:
        return self.ccy

    @property
    def start_date(self) -> date:
        if not self.cashflow_schedule:
            raise ValueError("Empty cashflow schedule")
        return min(d for d, _ in self.cashflow_schedule)

    @property
    def end_date(self) -> date:
        if not self.cashflow_schedule:
            raise ValueError("Empty cashflow schedule")
        return max(d for d, _ in self.cashflow_schedule)

    @property
    def notional(self) -> float:
        return sum(abs(a) for _, a in self.cashflow_schedule)

    @property
    def periods(self) -> PeriodList:
        return self._periods

    def cashflows(self, curve: Optional[Curve] = None) -> List[tuple[date, float]]:
        return list(self.cashflow_schedule)

    def npv(
        self,
        discount_curve: Curve,
        forecast_curve: Optional[Curve] = None,
        valuation_date: Optional[date] = None
    ) -> float:
        return sum(p.npv(discount_curve, forecast_curve) for p in self._periods)

    @classmethod
    def from_dataframe(
        cls,
        df,  # polars DataFrame with 'date' and 'amount' columns
        ccy: Union[str, Currency],
        description: str = "Custom"
    ) -> CustomLeg:
        """Create CustomLeg from Polars DataFrame."""
        cashflows = [
            (row['date'], row['amount'])
            for row in df.iter_rows(named=True)
        ]
        return cls(cashflow_schedule=cashflows, ccy=ccy, description=description)

    def __repr__(self) -> str:
        return f"CustomLeg({len(self.cashflow_schedule)} cashflows, {self.ccy.value})"


# =============================================================================
# Amortizing Leg
# =============================================================================

@dataclass
class AmortizingLeg(BaseLeg):
    """
    Fixed or floating leg with amortizing notional.

    Combines coupon payments with principal amortization.

    Args:
        effective_date: Start date
        termination_date: End date
        initial_notional: Starting notional
        amortization_schedule: Schedule of (date, remaining_notional)
        rate: Fixed rate (None for floating)
        ccy: Currency
        frequency: Payment frequency
        day_count: Day count convention
    """
    effective_date: date
    termination_date: date
    initial_notional: float
    amortization_schedule: List[tuple[date, float]]
    ccy: Currency
    rate: Optional[Rate] = None
    spread: Rate = field(default_factory=lambda: Rate(0.0))
    frequency: str = "3M"
    day_count: str = "ACT/360"
    pay_receive: int = 1
    include_principal: bool = True
    _periods: PeriodList = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.rate, (float, int, str)):
            self.rate = Rate(self.rate)
        if isinstance(self.spread, (float, int, str)):
            self.spread = Rate(self.spread)
        if isinstance(self.ccy, str):
            self.ccy = Currency.from_string(self.ccy)
        self._build_periods()

    def _build_periods(self) -> None:
        """Build amortizing periods with principal exchanges."""
        self._periods = []

        # Create notional lookup
        notional_by_date = dict(self.amortization_schedule)

        # Generate schedule
        schedule_dates = generate_schedule_dates(
            self.effective_date, self.termination_date, self.frequency
        )

        prev_notional = self.initial_notional
        for i in range(len(schedule_dates) - 1):
            start = schedule_dates[i]
            end = schedule_dates[i + 1]

            # Get notional for this period
            current_notional = notional_by_date.get(start, prev_notional)

            # Create coupon period
            if self.rate is not None:
                period = FixedCouponPeriod(
                    start=start,
                    end=end,
                    payment=end,
                    rate=self.rate,
                    notional_amount=current_notional * self.pay_receive,
                    ccy=self.ccy,
                    day_count=self.day_count
                )
            else:
                period = FloatCouponPeriod(
                    start=start,
                    end=end,
                    payment=end,
                    fixing_date=start,
                    notional_amount=current_notional * self.pay_receive,
                    ccy=self.ccy,
                    spread=self.spread,
                    day_count=self.day_count
                )
            self._periods.append(period)

            # Add principal exchange if notional changed and include_principal
            if self.include_principal and current_notional != prev_notional:
                principal = PrincipalExchange(
                    exchange_date=start,
                    amount=abs(prev_notional - current_notional),
                    ccy=self.ccy,
                    exchange_type=ExchangeType.INTERMEDIATE,
                    pay_receive=-self.pay_receive  # Principal repayment is opposite
                )
                self._periods.append(principal)

            prev_notional = current_notional

    @property
    def currency(self) -> Currency:
        return self.ccy

    @property
    def start_date(self) -> date:
        return self.effective_date

    @property
    def end_date(self) -> date:
        return self.termination_date

    @property
    def notional(self) -> float:
        return self.initial_notional

    @property
    def periods(self) -> PeriodList:
        return self._periods

    def cashflows(self, curve: Optional[Curve] = None) -> List[tuple[date, float]]:
        return [(p.payment_date, p.cashflow(curve)) for p in self._periods]

    def npv(
        self,
        discount_curve: Curve,
        forecast_curve: Optional[Curve] = None,
        valuation_date: Optional[date] = None
    ) -> float:
        fc = forecast_curve if forecast_curve is not None else discount_curve
        return sum(p.npv(discount_curve, fc) for p in self._periods)

    def __repr__(self) -> str:
        rate_str = str(self.rate) if self.rate else "floating"
        return (
            f"AmortizingLeg({rate_str}, "
            f"initial={self.initial_notional:,.0f}, "
            f"{len(self.amortization_schedule)} amort steps)"
        )


# =============================================================================
# Credit Premium Leg (for CDS)
# =============================================================================

@dataclass
class CreditPremiumLeg(BaseLeg):
    """
    CDS premium payment leg.

    Pays periodic premium until default or maturity.

    Args:
        effective_date: Protection start date
        termination_date: Protection end date
        spread: CDS spread (premium)
        notional_amount: Protected notional
        ccy: Currency
        frequency: Premium payment frequency
    """
    effective_date: date
    termination_date: date
    premium_spread: Rate
    notional_amount: float
    ccy: Currency
    frequency: str = "3M"
    day_count: str = "ACT/360"
    pay_receive: int = -1  # Typically pay premium
    _periods: PeriodList = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.premium_spread, (float, int, str)):
            self.premium_spread = Rate(self.premium_spread)
        if isinstance(self.ccy, str):
            self.ccy = Currency.from_string(self.ccy)
        self._build_periods()

    def _build_periods(self) -> None:
        """Build premium payment periods."""
        self._periods = []
        schedule_dates = generate_schedule_dates(
            self.effective_date, self.termination_date, self.frequency
        )

        for i in range(len(schedule_dates) - 1):
            start = schedule_dates[i]
            end = schedule_dates[i + 1]

            period = FixedCouponPeriod(
                start=start,
                end=end,
                payment=end,
                rate=self.premium_spread,
                notional_amount=self.notional_amount * self.pay_receive,
                ccy=self.ccy,
                day_count=self.day_count
            )
            self._periods.append(period)

    @property
    def currency(self) -> Currency:
        return self.ccy

    @property
    def start_date(self) -> date:
        return self.effective_date

    @property
    def end_date(self) -> date:
        return self.termination_date

    @property
    def notional(self) -> float:
        return self.notional_amount

    @property
    def periods(self) -> PeriodList:
        return self._periods

    def cashflows(self, curve: Optional[Curve] = None) -> List[tuple[date, float]]:
        return [(p.payment_date, p.cashflow(curve)) for p in self._periods]

    def risky_annuity(
        self,
        discount_curve: Curve,
        survival_curve: Curve
    ) -> float:
        """Calculate risky annuity (survival-weighted PV of 1bp)."""
        annuity = 0.0
        for period in self._periods:
            dcf = period.accrual_fraction()
            df = discount_curve.discount_factor(period.payment_date)
            # Get survival probability at payment date
            surv = survival_curve.discount_factor(period.payment_date)
            annuity += dcf * df * surv * self.notional_amount
        return annuity / 10000.0  # Per bp

    def npv(
        self,
        discount_curve: Curve,
        forecast_curve: Optional[Curve] = None,
        valuation_date: Optional[date] = None
    ) -> float:
        return sum(p.npv(discount_curve) for p in self._periods)

    def __repr__(self) -> str:
        return (
            f"CreditPremiumLeg({self.premium_spread}, "
            f"{self.effective_date} -> {self.termination_date})"
        )
