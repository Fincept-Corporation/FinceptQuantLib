"""
Day Count Conventions

Implements standard day count conventions used in fixed income markets:
- ACT/360: Actual days / 360 (money market)
- ACT/365F: Actual days / 365 Fixed
- ACT/365A: Actual days / 365 or 366 (leap year aware)
- ACT/ACT ISDA: Actual/Actual (ISDA method)
- 30/360: 30 days per month / 360 (US bond basis)
- 30E/360: 30E/360 European
- 30E/360 ISDA: ISDA variant
- BUS/252: Business days / 252 (Brazilian market)

All conventions follow ISDA 2006 definitions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, timedelta
from typing import Union

from finceptquantlib.scheduling.calendar import Calendar, WeekendCalendar


class DayCountConvention(ABC):
    """
    Abstract base class for day count conventions.

    Day count conventions determine how to calculate the fraction of a year
    between two dates for interest accrual and discounting purposes.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the standard name of the convention."""
        pass

    @abstractmethod
    def day_count(self, start: date, end: date) -> int:
        """
        Calculate the number of days between two dates.

        Args:
            start: Start date (inclusive)
            end: End date (exclusive)

        Returns:
            Number of days according to this convention
        """
        pass

    @abstractmethod
    def year_fraction(self, start: date, end: date) -> float:
        """
        Calculate the year fraction between two dates.

        Args:
            start: Start date
            end: End date

        Returns:
            Fraction of a year between the dates
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return self.name


class Actual360(DayCountConvention):
    """
    Actual/360 day count convention.

    The number of days between dates is the actual number of calendar days,
    divided by 360. Commonly used for money market instruments and
    floating rate notes.

    Examples:
        >>> dc = Actual360()
        >>> dc.year_fraction(date(2024, 1, 1), date(2024, 7, 1))  # 182/360
        0.5055555555555555
    """

    @property
    def name(self) -> str:
        return "ACT/360"

    def day_count(self, start: date, end: date) -> int:
        return (end - start).days

    def year_fraction(self, start: date, end: date) -> float:
        return self.day_count(start, end) / 360.0


class Actual365Fixed(DayCountConvention):
    """
    Actual/365 Fixed day count convention.

    The number of actual calendar days divided by 365 (ignoring leap years).
    Common in UK, Canada, Australia markets.

    Examples:
        >>> dc = Actual365Fixed()
        >>> dc.year_fraction(date(2024, 1, 1), date(2025, 1, 1))  # 366/365 (leap year)
        1.0027397260273974
    """

    @property
    def name(self) -> str:
        return "ACT/365F"

    def day_count(self, start: date, end: date) -> int:
        return (end - start).days

    def year_fraction(self, start: date, end: date) -> float:
        return self.day_count(start, end) / 365.0


class Actual365Actual(DayCountConvention):
    """
    Actual/365 Actual (or Actual/365L) day count convention.

    Uses 365 or 366 depending on whether the period includes Feb 29.
    Used in some markets for floating rate notes.
    """

    @property
    def name(self) -> str:
        return "ACT/365A"

    def day_count(self, start: date, end: date) -> int:
        return (end - start).days

    def _is_leap_year(self, year: int) -> bool:
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    def _contains_feb29(self, start: date, end: date) -> bool:
        """Check if the period [start, end) contains February 29."""
        for year in range(start.year, end.year + 1):
            if self._is_leap_year(year):
                feb29 = date(year, 2, 29)
                if start <= feb29 < end:
                    return True
        return False

    def year_fraction(self, start: date, end: date) -> float:
        days = self.day_count(start, end)
        denominator = 366.0 if self._contains_feb29(start, end) else 365.0
        return days / denominator


class ActualActualISDA(DayCountConvention):
    """
    Actual/Actual ISDA day count convention.

    The most accurate convention - counts actual days in each year,
    dividing by the actual number of days in that year.

    For periods spanning multiple years, the fraction is computed
    for each year segment and summed.

    Examples:
        >>> dc = ActualActualISDA()
        >>> dc.year_fraction(date(2024, 1, 1), date(2025, 1, 1))  # Exactly one year
        1.0
    """

    @property
    def name(self) -> str:
        return "ACT/ACT ISDA"

    def day_count(self, start: date, end: date) -> int:
        return (end - start).days

    def _days_in_year(self, year: int) -> int:
        return 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365

    def year_fraction(self, start: date, end: date) -> float:
        if start >= end:
            return 0.0

        if start.year == end.year:
            return self.day_count(start, end) / self._days_in_year(start.year)

        # Split across years
        fraction = 0.0

        # First year: from start to end of year
        year_end = date(start.year + 1, 1, 1)
        fraction += (year_end - start).days / self._days_in_year(start.year)

        # Full years in between
        for year in range(start.year + 1, end.year):
            fraction += 1.0

        # Last year: from start of year to end
        if end > date(end.year, 1, 1):
            year_start = date(end.year, 1, 1)
            fraction += (end - year_start).days / self._days_in_year(end.year)

        return fraction


class Thirty360(DayCountConvention):
    """
    30/360 day count convention (US Bond Basis).

    Assumes 30 days per month and 360 days per year.
    Special rules apply at month ends.

    ISDA 2006 definition (US variant):
    - If D1 is 31, change D1 to 30
    - If D2 is 31 and D1 is 30 or 31, change D2 to 30

    Examples:
        >>> dc = Thirty360()
        >>> dc.year_fraction(date(2024, 1, 31), date(2024, 2, 28))  # D1 31 -> 30: 28/360
        0.07777777777777778
    """

    @property
    def name(self) -> str:
        return "30/360"

    @staticmethod
    def _is_last_day_of_feb(d: date) -> bool:
        return d.month == 2 and d.day == (29 if d.year % 4 == 0 and (d.year % 100 != 0 or d.year % 400 == 0) else 28)

    def day_count(self, start: date, end: date) -> int:
        d1, m1, y1 = start.day, start.month, start.year
        d2, m2, y2 = end.day, end.month, end.year

        # Apply 30/360 US rules (ISDA 2006)
        if self._is_last_day_of_feb(start):
            d1 = 30
        if d1 == 31:
            d1 = 30
        if self._is_last_day_of_feb(end) and self._is_last_day_of_feb(start):
            d2 = 30
        if d2 == 31 and d1 >= 30:
            d2 = 30

        return 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)

    def year_fraction(self, start: date, end: date) -> float:
        return self.day_count(start, end) / 360.0


class Thirty360European(DayCountConvention):
    """
    30/360 European (30E/360) day count convention.

    Also known as Eurobond basis or ISMA 30/360.

    Rules:
    - If D1 is 31, change D1 to 30
    - If D2 is 31, change D2 to 30
    """

    @property
    def name(self) -> str:
        return "30E/360"

    def day_count(self, start: date, end: date) -> int:
        d1, m1, y1 = start.day, start.month, start.year
        d2, m2, y2 = end.day, end.month, end.year

        if d1 == 31:
            d1 = 30
        if d2 == 31:
            d2 = 30

        return 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)

    def year_fraction(self, start: date, end: date) -> float:
        return self.day_count(start, end) / 360.0


# Alias
ThirtyE360 = Thirty360European


class ThirtyE360ISDA(DayCountConvention):
    """
    30E/360 ISDA day count convention.

    Modified European 30/360 with special treatment for February.

    Rules:
    - If D1 is last day of month, change D1 to 30
    - If D2 is last day of month (unless Feb and end is maturity), change D2 to 30
    """

    def __init__(self, maturity_date: date | None = None):
        self._maturity_date = maturity_date

    @property
    def name(self) -> str:
        return "30E/360 ISDA"

    def _is_last_day_of_month(self, d: date) -> bool:
        next_day = d + timedelta(days=1)
        return next_day.month != d.month

    def _is_last_day_of_february(self, d: date) -> bool:
        return d.month == 2 and self._is_last_day_of_month(d)

    def day_count(self, start: date, end: date) -> int:
        d1, m1, y1 = start.day, start.month, start.year
        d2, m2, y2 = end.day, end.month, end.year

        if self._is_last_day_of_month(start):
            d1 = 30

        if self._is_last_day_of_month(end):
            # Exception for February end being maturity
            if not (self._is_last_day_of_february(end) and
                    self._maturity_date and end == self._maturity_date):
                d2 = 30

        return 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)

    def year_fraction(self, start: date, end: date) -> float:
        return self.day_count(start, end) / 360.0


class Business252(DayCountConvention):
    """
    Business/252 day count convention.

    Used primarily in Brazilian markets. Counts business days
    (excluding weekends and holidays) and divides by 252.

    Requires a calendar to determine business days.
    """

    def __init__(self, calendar: "Calendar | None" = None):
        self._calendar = calendar or WeekendCalendar()

    @property
    def name(self) -> str:
        return "BUS/252"

    def day_count(self, start: date, end: date) -> int:
        """Count business days between dates."""
        count = 0
        current = start
        while current < end:
            if self._calendar.is_business_day(current):
                count += 1
            current += timedelta(days=1)
        return count

    def year_fraction(self, start: date, end: date) -> float:
        return self.day_count(start, end) / 252.0


# Convenience function

def year_fraction(
    start: date,
    end: date,
    convention: Union[str, DayCountConvention] = "ACT/360"
) -> float:
    """
    Calculate year fraction between two dates.

    Args:
        start: Start date
        end: End date
        convention: Day count convention (string or object)

    Returns:
        Year fraction

    Examples:
        >>> year_fraction(date(2024, 1, 1), date(2024, 7, 1), "ACT/360")
        0.5055555555555555

        >>> year_fraction(date(2024, 1, 1), date(2024, 7, 1), "30/360")
        0.5
    """
    if isinstance(convention, str):
        convention = _get_convention(convention)
    return convention.year_fraction(start, end)


_CONVENTION_CACHE: dict[str, DayCountConvention] = {}

def _get_convention(name: str) -> DayCountConvention:
    """Get day count convention by name (cached)."""
    key = name.upper().replace("_", "/").replace("-", "/")

    if key in _CONVENTION_CACHE:
        return _CONVENTION_CACHE[key]

    _CONVENTIONS_MAP = {
        "ACT/360": Actual360,
        "ACTUAL/360": Actual360,
        "A/360": Actual360,
        "ACT/365": Actual365Fixed,
        "ACT/365F": Actual365Fixed,
        "ACT/365 FIXED": Actual365Fixed,
        "ACTUAL/365": Actual365Fixed,
        "ACT/365A": Actual365Actual,
        "ACT/365 ACTUAL": Actual365Actual,
        "ACT/ACT": ActualActualISDA,
        "ACT/ACT ISDA": ActualActualISDA,
        "ACTUAL/ACTUAL": ActualActualISDA,
        "30/360": Thirty360,
        "30/360 US": Thirty360,
        "BOND BASIS": Thirty360,
        "30E/360": Thirty360European,
        "30/360 EUROPEAN": Thirty360European,
        "EUROBOND BASIS": Thirty360European,
        "30E/360 ISDA": ThirtyE360ISDA,
        "BUS/252": Business252,
        "BUSINESS/252": Business252,
    }

    if key not in _CONVENTIONS_MAP:
        valid = ", ".join(sorted(set(_CONVENTIONS_MAP.keys())))
        raise ValueError(f"Unknown day count convention: {name}. Valid: {valid}")

    instance = _CONVENTIONS_MAP[key]()
    _CONVENTION_CACHE[key] = instance
    return instance
