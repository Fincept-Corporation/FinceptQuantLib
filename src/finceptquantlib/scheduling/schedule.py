"""
Payment Schedule Generation

Generates payment schedules for fixed income instruments with support for:
- Various frequencies (annual, semi-annual, quarterly, monthly)
- Stub periods (front, back, short, long)
- Roll conventions (end of month, IMM dates)
- Business day adjustments
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from typing import List, Optional

from finceptquantlib.core.foundations.types import Frequency, StubType, RollConvention
from finceptquantlib.scheduling.calendar import Calendar, WeekendCalendar, get_calendar
from finceptquantlib.scheduling.adjustment import (
    BusinessDayAdjustment,
)
from finceptquantlib.scheduling.daycount import DayCountConvention


@dataclass
class SchedulePeriod:
    """
    A single period in a payment schedule.

    Attributes:
        start_date: Period start date (unadjusted)
        end_date: Period end date (unadjusted)
        payment_date: Payment date (adjusted)
        accrual_start: Accrual start date (may differ from start_date)
        accrual_end: Accrual end date (may differ from end_date)
        notional: Period notional (for amortizing schedules)
    """
    start_date: date
    end_date: date
    payment_date: date
    accrual_start: date
    accrual_end: date
    notional: float = 1.0

    @property
    def days(self) -> int:
        """Number of calendar days in the period."""
        return (self.accrual_end - self.accrual_start).days

    def year_fraction(
        self,
        convention: DayCountConvention | str = "ACT/360"
    ) -> float:
        """Calculate year fraction for this period."""
        from finceptquantlib.scheduling.daycount import year_fraction as yf
        return yf(self.accrual_start, self.accrual_end, convention)


class Schedule:
    """
    Payment schedule for a fixed income instrument.

    Generates a series of payment periods based on start/end dates,
    frequency, and various conventions.

    Attributes:
        effective_date: Schedule start date
        termination_date: Schedule end date
        frequency: Payment frequency
        calendar: Business day calendar
        adjustment: Business day adjustment rule
        stub: Stub period type
        roll_convention: Roll date convention
        periods: List of schedule periods
    """

    def __init__(
        self,
        effective_date: date,
        termination_date: date,
        frequency: Frequency | str = Frequency.SEMI_ANNUAL,
        calendar: Calendar | str = "Weekend",
        adjustment: BusinessDayAdjustment | str = "Modified Following",
        stub: StubType | str = StubType.SHORT_FRONT,
        roll_convention: RollConvention = RollConvention.NONE,
        roll_day: Optional[int] = None,
        first_coupon_date: Optional[date] = None,
        last_coupon_date: Optional[date] = None,
        payment_lag: int = 0,
    ):
        """
        Create a payment schedule.

        Args:
            effective_date: Start date of the schedule
            termination_date: End date of the schedule
            frequency: Payment frequency
            calendar: Business day calendar
            adjustment: Business day adjustment rule
            stub: Stub period handling
            roll_convention: Roll date convention
            roll_day: Specific day of month for rolling (1-31)
            first_coupon_date: First coupon date (for stub)
            last_coupon_date: Last coupon date (for stub)
            payment_lag: Days after period end for payment
        """
        self.effective_date = effective_date
        self.termination_date = termination_date

        # Parse frequency
        if isinstance(frequency, str):
            frequency = _parse_frequency(frequency)
        self.frequency = frequency

        # Parse calendar
        if isinstance(calendar, str):
            calendar = get_calendar(calendar)
        elif calendar is None:
            calendar = WeekendCalendar()
        self.calendar = calendar

        # Parse adjustment
        if isinstance(adjustment, str):
            from finceptquantlib.scheduling.adjustment import _get_adjustment
            adjustment = _get_adjustment(adjustment)
        self.adjustment = adjustment

        # Parse stub
        if isinstance(stub, str):
            stub = _parse_stub(stub)
        self.stub = stub

        self.roll_convention = roll_convention
        self.roll_day = roll_day
        self.first_coupon_date = first_coupon_date
        self.last_coupon_date = last_coupon_date
        self.payment_lag = payment_lag

        # Generate periods
        self.periods = self._generate_periods()

    def _generate_periods(self) -> List[SchedulePeriod]:
        """Generate the schedule periods."""
        if self.frequency == Frequency.ZERO_COUPON:
            return self._generate_zero_coupon()

        return self._generate_regular()

    def _generate_zero_coupon(self) -> List[SchedulePeriod]:
        """Generate a single period for zero-coupon instruments."""
        payment_date = self._adjust_date(self.termination_date)
        if self.payment_lag > 0:
            payment_date = self.calendar.add_business_days(
                payment_date, self.payment_lag
            )

        return [
            SchedulePeriod(
                start_date=self.effective_date,
                end_date=self.termination_date,
                payment_date=payment_date,
                accrual_start=self.effective_date,
                accrual_end=self.termination_date,
            )
        ]

    def _generate_regular(self) -> List[SchedulePeriod]:
        """Generate regular periodic schedule."""
        months = self.frequency.months

        # Generate unadjusted dates
        if months == 0:
            # WEEKLY or DAILY — use day-based generation
            dates = self._generate_day_based()
        elif self.stub in (StubType.SHORT_FRONT, StubType.LONG_FRONT):
            dates = self._generate_backward(months)
        else:
            dates = self._generate_forward(months)

        # Apply LONG stub merging
        if len(dates) >= 3:
            if self.stub == StubType.LONG_FRONT and (dates[1] - dates[0]).days < (dates[2] - dates[1]).days * 0.5:
                dates = [dates[0]] + dates[2:]
            elif self.stub == StubType.LONG_BACK and (dates[-1] - dates[-2]).days < (dates[-2] - dates[-3]).days * 0.5:
                dates = dates[:-2] + [dates[-1]]

        # Use first_coupon_date / last_coupon_date overrides
        if self.first_coupon_date and len(dates) >= 2:
            dates[1] = self.first_coupon_date
        if self.last_coupon_date and len(dates) >= 2:
            dates[-2] = self.last_coupon_date

        # Build periods
        periods = []
        for i in range(len(dates) - 1):
            start = dates[i]
            end = dates[i + 1]

            # Adjust payment date
            payment_date = self._adjust_date(end)
            if self.payment_lag > 0:
                payment_date = self.calendar.add_business_days(
                    payment_date, self.payment_lag
                )

            # Accrual dates might be adjusted differently
            accrual_start = self._adjust_date(start) if i == 0 else periods[-1].accrual_end
            accrual_end = self._adjust_date(end)

            periods.append(
                SchedulePeriod(
                    start_date=start,
                    end_date=end,
                    payment_date=payment_date,
                    accrual_start=accrual_start,
                    accrual_end=accrual_end,
                )
            )

        return periods

    def _generate_day_based(self) -> List[date]:
        """Generate dates for WEEKLY or DAILY frequencies."""
        if self.frequency == Frequency.WEEKLY:
            step = timedelta(weeks=1)
        else:
            step = timedelta(days=1)

        dates = [self.effective_date]
        current = self.effective_date
        while True:
            current = current + step
            if current >= self.termination_date:
                if dates[-1] != self.termination_date:
                    dates.append(self.termination_date)
                break
            dates.append(current)
        return dates

    def _generate_backward(self, months: int) -> List[date]:
        """Generate dates backward from termination date."""
        # Build in reverse, then flip — avoids O(n²) insert(0)
        reverse_dates = [self.termination_date]
        current = self.termination_date

        while current > self.effective_date:
            current = current - relativedelta(months=months)
            current = self._apply_roll_convention(current)

            if current > self.effective_date:
                reverse_dates.append(current)
            else:
                reverse_dates.append(self.effective_date)
                break

        # Ensure effective date is included
        if reverse_dates[-1] != self.effective_date:
            reverse_dates.append(self.effective_date)

        reverse_dates.reverse()
        return reverse_dates

    def _generate_forward(self, months: int) -> List[date]:
        """Generate dates forward from effective date."""
        dates = [self.effective_date]
        current = self.effective_date

        while current < self.termination_date:
            current = current + relativedelta(months=months)

            # Apply roll convention
            current = self._apply_roll_convention(current)

            if current <= self.termination_date:
                dates.append(current)
            else:
                if dates[-1] != self.termination_date:
                    dates.append(self.termination_date)
                break

        # Ensure termination date is last
        if dates[-1] != self.termination_date:
            dates.append(self.termination_date)

        return dates

    def _apply_roll_convention(self, d: date) -> date:
        """Apply roll convention to a date."""
        if self.roll_convention == RollConvention.EOM:
            # Roll to end of month
            if d.month == 12:
                next_month = date(d.year + 1, 1, 1)
            else:
                next_month = date(d.year, d.month + 1, 1)
            return next_month - timedelta(days=1)

        elif self.roll_convention == RollConvention.SPECIFIC_DAY and self.roll_day:
            # Roll to specific day
            try:
                return date(d.year, d.month, self.roll_day)
            except ValueError:
                # Day doesn't exist in month, use last day
                if d.month == 12:
                    next_month = date(d.year + 1, 1, 1)
                else:
                    next_month = date(d.year, d.month + 1, 1)
                return next_month - timedelta(days=1)

        elif self.roll_convention == RollConvention.IMM:
            # Roll to IMM date (3rd Wednesday)
            return self._get_imm_date(d.year, d.month)

        return d

    def _get_imm_date(self, year: int, month: int) -> date:
        """Get IMM date (3rd Wednesday) for a month."""
        first = date(year, month, 1)
        # Find first Wednesday
        wed_offset = (2 - first.weekday()) % 7
        first_wed = first + timedelta(days=wed_offset)
        # Third Wednesday
        return first_wed + timedelta(weeks=2)

    def _adjust_date(self, d: date) -> date:
        """Adjust a date using the calendar and adjustment rule."""
        return self.adjustment.adjust(d, self.calendar)

    def __len__(self) -> int:
        return len(self.periods)

    def __getitem__(self, index: int) -> SchedulePeriod:
        return self.periods[index]

    def __iter__(self):
        return iter(self.periods)

    @property
    def dates(self) -> List[date]:
        """Get all period end dates."""
        if not self.periods:
            return []
        return [self.periods[0].start_date] + [p.end_date for p in self.periods]

    @property
    def payment_dates(self) -> List[date]:
        """Get all payment dates."""
        return [p.payment_date for p in self.periods]


def generate_schedule(
    effective_date: date,
    termination_date: date,
    frequency: str = "Semi-Annual",
    calendar: str = "Weekend",
    adjustment: str = "Modified Following",
    stub: str = "Short Front",
) -> Schedule:
    """
    Generate a payment schedule.

    Convenience function for creating schedules.

    Args:
        effective_date: Start date
        termination_date: End date
        frequency: Payment frequency
        calendar: Business day calendar
        adjustment: Business day adjustment
        stub: Stub period type

    Returns:
        Schedule object

    Examples:
        >>> schedule = generate_schedule(
        ...     effective_date=date(2024, 1, 15),
        ...     termination_date=date(2034, 1, 15),
        ...     frequency="Semi-Annual",
        ...     calendar="US",
        ...     adjustment="Modified Following"
        ... )
        >>> len(schedule)  # 10 years * 2 periods per year
        20
    """
    return Schedule(
        effective_date=effective_date,
        termination_date=termination_date,
        frequency=frequency,
        calendar=calendar,
        adjustment=adjustment,
        stub=stub,
    )


def _parse_frequency(freq: str) -> Frequency:
    """Parse frequency string to Frequency enum."""
    mapping = {
        "ANNUAL": Frequency.ANNUAL,
        "A": Frequency.ANNUAL,
        "1Y": Frequency.ANNUAL,
        "SEMI-ANNUAL": Frequency.SEMI_ANNUAL,
        "SEMIANNUAL": Frequency.SEMI_ANNUAL,
        "S": Frequency.SEMI_ANNUAL,
        "SA": Frequency.SEMI_ANNUAL,
        "6M": Frequency.SEMI_ANNUAL,
        "QUARTERLY": Frequency.QUARTERLY,
        "Q": Frequency.QUARTERLY,
        "3M": Frequency.QUARTERLY,
        "MONTHLY": Frequency.MONTHLY,
        "M": Frequency.MONTHLY,
        "1M": Frequency.MONTHLY,
        "WEEKLY": Frequency.WEEKLY,
        "W": Frequency.WEEKLY,
        "DAILY": Frequency.DAILY,
        "D": Frequency.DAILY,
        "ZERO": Frequency.ZERO_COUPON,
        "ZERO COUPON": Frequency.ZERO_COUPON,
        "Z": Frequency.ZERO_COUPON,
    }

    key = freq.upper().strip().replace("_", "-")
    if key not in mapping:
        raise ValueError(f"Unknown frequency: {freq}")
    return mapping[key]


def _parse_stub(stub: str) -> StubType:
    """Parse stub type string."""
    mapping = {
        "NONE": StubType.NONE,
        "SHORT FRONT": StubType.SHORT_FRONT,
        "SHORTFRONT": StubType.SHORT_FRONT,
        "SF": StubType.SHORT_FRONT,
        "LONG FRONT": StubType.LONG_FRONT,
        "LONGFRONT": StubType.LONG_FRONT,
        "LF": StubType.LONG_FRONT,
        "SHORT BACK": StubType.SHORT_BACK,
        "SHORTBACK": StubType.SHORT_BACK,
        "SB": StubType.SHORT_BACK,
        "LONG BACK": StubType.LONG_BACK,
        "LONGBACK": StubType.LONG_BACK,
        "LB": StubType.LONG_BACK,
    }

    key = stub.upper().strip().replace("_", " ").replace("-", " ")
    if key not in mapping:
        raise ValueError(f"Unknown stub type: {stub}")
    return mapping[key]
