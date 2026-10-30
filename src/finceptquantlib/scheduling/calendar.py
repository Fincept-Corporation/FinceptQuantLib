"""
Business Day Calendars

Implements calendar systems for major financial centers:
- Weekend-only calendar
- TARGET (Trans-European Automated Real-time Gross Settlement Express Transfer)
- US Federal Reserve
- UK (London)
- Japan (Tokyo)
- Combined calendars

Calendars are used to determine business days and holidays for
payment date adjustments and accrual calculations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, timedelta
from functools import lru_cache
from typing import List


# =============================================================================
# Shared calendar utilities
# =============================================================================

@lru_cache(maxsize=200)
def _easter_date(year: int) -> date:
    """Calculate Easter Sunday using the Anonymous Gregorian algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    el = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * el) // 451
    month = (h + el - 7 * m + 114) // 31
    day = ((h + el - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Find the nth occurrence of a weekday in a month."""
    first_day = date(year, month, 1)
    first_weekday = first_day + timedelta(days=(weekday - first_day.weekday()) % 7)
    return first_weekday + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Find the last occurrence of a weekday in a month."""
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    last_day = next_month - timedelta(days=1)
    offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=offset)


def _observe_weekend_us(d: date) -> date:
    """Adjust for US weekend observance (Friday if Saturday, Monday if Sunday)."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    elif d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _substitute_holiday_uk(d: date) -> date:
    """Move UK weekend holidays to next working day."""
    if d.weekday() == 5:
        return d + timedelta(days=2)
    elif d.weekday() == 6:
        return d + timedelta(days=1)
    return d


class Calendar(ABC):
    """
    Abstract base class for business day calendars.

    A calendar defines which days are business days (trading days)
    and which are holidays.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the calendar name."""
        pass

    @abstractmethod
    def is_holiday(self, d: date) -> bool:
        """
        Check if a date is a holiday.

        Args:
            d: Date to check

        Returns:
            True if the date is a holiday
        """
        pass

    def is_weekend(self, d: date) -> bool:
        """Check if a date is a weekend (Saturday or Sunday)."""
        return d.weekday() >= 5

    def is_business_day(self, d: date) -> bool:
        """
        Check if a date is a business day.

        A business day is not a weekend and not a holiday.

        Args:
            d: Date to check

        Returns:
            True if the date is a business day
        """
        return not self.is_weekend(d) and not self.is_holiday(d)

    def next_business_day(self, d: date) -> date:
        """
        Find the next business day on or after the given date.

        Args:
            d: Starting date

        Returns:
            Next business day
        """
        while not self.is_business_day(d):
            d += timedelta(days=1)
        return d

    def previous_business_day(self, d: date) -> date:
        """
        Find the previous business day on or before the given date.

        Args:
            d: Starting date

        Returns:
            Previous business day
        """
        while not self.is_business_day(d):
            d -= timedelta(days=1)
        return d

    def add_business_days(self, d: date, days: int) -> date:
        """
        Add a number of business days to a date.

        Args:
            d: Starting date
            days: Number of business days to add (can be negative)

        Returns:
            Resulting date
        """
        if days == 0:
            return d

        step = 1 if days > 0 else -1
        remaining = abs(days)

        while remaining > 0:
            d += timedelta(days=step)
            if self.is_business_day(d):
                remaining -= 1

        return d

    def business_days_between(self, start: date, end: date) -> int:
        """
        Count business days between two dates.

        Uses bulk weekend skipping for performance on long ranges.

        Args:
            start: Start date (inclusive)
            end: End date (exclusive)

        Returns:
            Number of business days
        """
        if start >= end:
            return 0

        total_days = (end - start).days

        # Bulk estimate: 5 business days per 7 calendar days
        full_weeks = total_days // 7
        count = full_weeks * 5

        # Count remaining partial week days
        remainder_start = start + timedelta(days=full_weeks * 7)
        current = remainder_start
        while current < end:
            if not self.is_weekend(current):
                count += 1
            current += timedelta(days=1)

        # Subtract holidays that fall on weekdays
        current = start
        while current < end:
            if not self.is_weekend(current) and self.is_holiday(current):
                count -= 1
            current += timedelta(days=1)

        return count

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return self.name


class WeekendCalendar(Calendar):
    """
    Simple weekend-only calendar.

    Treats all weekends (Saturday and Sunday) as non-business days.
    No other holidays are recognized.
    """

    @property
    def name(self) -> str:
        return "Weekend"

    def is_holiday(self, d: date) -> bool:
        return False  # No holidays, only weekends


class TargetCalendar(Calendar):
    """
    TARGET calendar (Trans-European Automated Real-time Gross Settlement).

    Used for EUR payments and settlements.

    Holidays:
    - New Year's Day (Jan 1)
    - Good Friday
    - Easter Monday
    - Labour Day (May 1)
    - Christmas Day (Dec 25)
    - Boxing Day (Dec 26)
    """

    @property
    def name(self) -> str:
        return "TARGET"

    def is_holiday(self, d: date) -> bool:
        return d in self._holidays_for_year(d.year)

    @classmethod
    @lru_cache(maxsize=256)
    def _holidays_for_year(cls, year: int) -> frozenset:
        """Compute all TARGET holidays for a year."""
        easter = _easter_date(year)
        holidays = {
            date(year, 1, 1),      # New Year's Day
            date(year, 5, 1),      # Labour Day
            date(year, 12, 25),    # Christmas Day
            date(year, 12, 26),    # Boxing Day
            easter - timedelta(days=2),  # Good Friday
            easter + timedelta(days=1),  # Easter Monday
        }
        return frozenset(holidays)


class USFederalCalendar(Calendar):
    """
    US Federal Reserve calendar.

    Holidays (if falls on weekend, observed on Monday/Friday):
    - New Year's Day (Jan 1)
    - Martin Luther King Jr. Day (3rd Monday in January)
    - Presidents' Day (3rd Monday in February)
    - Memorial Day (Last Monday in May)
    - Juneteenth (June 19, since 2021)
    - Independence Day (July 4)
    - Labor Day (1st Monday in September)
    - Columbus Day (2nd Monday in October)
    - Veterans Day (November 11)
    - Thanksgiving (4th Thursday in November)
    - Christmas Day (December 25)
    """

    @property
    def name(self) -> str:
        return "US Federal"

    def is_holiday(self, d: date) -> bool:
        return d in self._holidays_for_year(d.year)

    @classmethod
    @lru_cache(maxsize=256)
    def _holidays_for_year(cls, year: int) -> frozenset:
        """Compute all US Federal holidays for a year."""
        holidays = {
            _observe_weekend_us(date(year, 1, 1)),       # New Year's Day
            _nth_weekday(year, 1, 0, 3),                 # MLK Day
            _nth_weekday(year, 2, 0, 3),                 # Presidents' Day
            _last_weekday(year, 5, 0),                    # Memorial Day
            _observe_weekend_us(date(year, 7, 4)),       # Independence Day
            _nth_weekday(year, 9, 0, 1),                 # Labor Day
            _nth_weekday(year, 10, 0, 2),                # Columbus Day
            _observe_weekend_us(date(year, 11, 11)),     # Veterans Day
            _nth_weekday(year, 11, 3, 4),                # Thanksgiving
            _observe_weekend_us(date(year, 12, 25)),     # Christmas Day
        }
        # Juneteenth (since 2021)
        if year >= 2021:
            holidays.add(_observe_weekend_us(date(year, 6, 19)))
        # A Saturday New Year's Day is observed on Friday Dec 31 of this year
        next_new_year = _observe_weekend_us(date(year + 1, 1, 1))
        if next_new_year.year == year:
            holidays.add(next_new_year)
        return frozenset(h for h in holidays if h.year == year)


# Bank holidays moved or added by royal proclamation
_UK_EARLY_MAY_MOVED = {
    1995: date(1995, 5, 8),    # VE Day 50th anniversary
    2020: date(2020, 5, 8),    # VE Day 75th anniversary
}
_UK_SPRING_MOVED = {
    2002: date(2002, 6, 4),    # Golden Jubilee
    2012: date(2012, 6, 4),    # Diamond Jubilee
    2022: date(2022, 6, 2),    # Platinum Jubilee
}
_UK_ONE_OFF = {
    1999: (date(1999, 12, 31),),                     # Millennium
    2002: (date(2002, 6, 3),),                       # Golden Jubilee
    2011: (date(2011, 4, 29),),                      # Royal Wedding
    2012: (date(2012, 6, 5),),                       # Diamond Jubilee
    2022: (date(2022, 6, 3), date(2022, 9, 19)),     # Platinum Jubilee, State Funeral
    2023: (date(2023, 5, 8),),                       # Coronation
}


class UKCalendar(Calendar):
    """
    UK (London) calendar.

    Holidays:
    - New Year's Day (Jan 1)
    - Good Friday
    - Easter Monday
    - Early May Bank Holiday (1st Monday in May)
    - Spring Bank Holiday (Last Monday in May)
    - Summer Bank Holiday (Last Monday in August)
    - Christmas Day (Dec 25)
    - Boxing Day (Dec 26)
    """

    @property
    def name(self) -> str:
        return "UK"

    def is_holiday(self, d: date) -> bool:
        return d in self._holidays_for_year(d.year)

    @classmethod
    @lru_cache(maxsize=256)
    def _holidays_for_year(cls, year: int) -> frozenset:
        """Compute all UK holidays for a year."""
        easter = _easter_date(year)

        # Boxing Day adjustment
        boxing = date(year, 12, 26)
        xmas_day = date(year, 12, 25)
        if boxing.weekday() == 5:       # Saturday -> Monday Dec 28
            boxing = date(year, 12, 28)
        elif boxing.weekday() == 6:     # Sunday -> Tuesday Dec 28
            boxing = date(year, 12, 28)
        elif boxing.weekday() == 0 and xmas_day.weekday() == 6:
            # Xmas is Sunday, substituted to Monday Dec 26, so Boxing -> Tue Dec 27
            boxing = date(year, 12, 27)

        holidays = {
            _substitute_holiday_uk(date(year, 1, 1)),     # New Year's Day
            easter - timedelta(days=2),                    # Good Friday
            easter + timedelta(days=1),                    # Easter Monday
            _UK_EARLY_MAY_MOVED.get(year, _nth_weekday(year, 5, 0, 1)),  # Early May
            _UK_SPRING_MOVED.get(year, _last_weekday(year, 5, 0)),       # Spring
            _last_weekday(year, 8, 0),                     # Summer Bank Holiday
            _substitute_holiday_uk(xmas_day),              # Christmas Day
            boxing,                                        # Boxing Day
        }
        holidays.update(_UK_ONE_OFF.get(year, ()))
        return frozenset(holidays)


class JapanCalendar(Calendar):
    """
    Japan (Tokyo) calendar.

    Key holidays (subset):
    - New Year's Day (Jan 1-3)
    - Coming of Age Day (2nd Monday in January)
    - National Foundation Day (Feb 11)
    - Emperor's Birthday (Feb 23)
    - Vernal Equinox Day (~Mar 20-21)
    - Showa Day (Apr 29)
    - Constitution Day (May 3)
    - Greenery Day (May 4)
    - Children's Day (May 5)
    - Marine Day (3rd Monday in July)
    - Mountain Day (Aug 11)
    - Respect for the Aged Day (3rd Monday in September)
    - Autumn Equinox (~Sep 22-23)
    - Sports Day (2nd Monday in October)
    - Culture Day (Nov 3)
    - Labor Thanksgiving Day (Nov 23)
    """

    @property
    def name(self) -> str:
        return "Japan"

    @staticmethod
    def _vernal_equinox(year: int) -> int:
        """Approximate vernal equinox day (March) using astronomical formula."""
        if year <= 1947:
            return 21
        elif year <= 1979:
            return int(20.8357 + 0.242194 * (year - 1980)) - int((year - 1983) // 4)
        elif year <= 2099:
            return int(20.8431 + 0.242194 * (year - 1980)) - int((year - 1980) // 4)
        else:
            return int(21.8510 + 0.242194 * (year - 1980)) - int((year - 1980) // 4)

    @staticmethod
    def _autumn_equinox(year: int) -> int:
        """Approximate autumn equinox day (September) using astronomical formula."""
        if year <= 1947:
            return 23
        elif year <= 1979:
            return int(23.2588 + 0.242194 * (year - 1980)) - int((year - 1983) // 4)
        elif year <= 2099:
            return int(23.2488 + 0.242194 * (year - 1980)) - int((year - 1980) // 4)
        else:
            return int(24.2488 + 0.242194 * (year - 1980)) - int((year - 1980) // 4)

    @staticmethod
    def _sub_jp(d: date) -> date:
        """Move Sunday holidays to Monday (Japan substitute holiday rule)."""
        if d.weekday() == 6:
            return d + timedelta(days=1)
        return d

    def is_holiday(self, d: date) -> bool:
        return d in self._holidays_for_year(d.year)

    @classmethod
    @lru_cache(maxsize=256)
    def _holidays_for_year(cls, year: int) -> frozenset:
        """Compute all Japan holidays for a year."""
        sub = cls._sub_jp
        holidays = {
            date(year, 1, 1),                  # New Year's Day
            date(year, 1, 2),                  # Jan 2
            date(year, 1, 3),                  # Jan 3
            _nth_weekday(year, 1, 0, 2),       # Coming of Age Day
            sub(date(year, 2, 11)),            # National Foundation Day
            sub(date(year, 2, 23)),            # Emperor's Birthday
            sub(date(year, 3, cls._vernal_equinox(year))),  # Vernal Equinox
            sub(date(year, 4, 29)),            # Showa Day
            sub(date(year, 5, 3)),             # Constitution Day
            sub(date(year, 5, 4)),             # Greenery Day
            sub(date(year, 5, 5)),             # Children's Day
            _nth_weekday(year, 7, 0, 3),       # Marine Day
            sub(date(year, 8, 11)),            # Mountain Day
            _nth_weekday(year, 9, 0, 3),       # Respect for Aged Day
            sub(date(year, 9, cls._autumn_equinox(year))),  # Autumn Equinox
            _nth_weekday(year, 10, 0, 2),      # Sports Day
            sub(date(year, 11, 3)),            # Culture Day
            sub(date(year, 11, 23)),           # Labor Thanksgiving Day
        }
        return frozenset(holidays)


class CombinedCalendar(Calendar):
    """
    Combined calendar from multiple calendars.

    A day is a holiday if it's a holiday in ANY of the combined calendars.
    Useful for cross-currency trades.
    """

    def __init__(self, calendars: List[Calendar]):
        if not calendars:
            raise ValueError("At least one calendar required")
        self._calendars = calendars

    @property
    def name(self) -> str:
        return " + ".join(c.name for c in self._calendars)

    def is_holiday(self, d: date) -> bool:
        return any(cal.is_holiday(d) for cal in self._calendars)


# Calendar registry

_CALENDARS = {
    "WEEKEND": WeekendCalendar,
    "TARGET": TargetCalendar,
    "EUR": TargetCalendar,
    "US": USFederalCalendar,
    "USD": USFederalCalendar,
    "FED": USFederalCalendar,
    "UK": UKCalendar,
    "GBP": UKCalendar,
    "LONDON": UKCalendar,
    "JAPAN": JapanCalendar,
    "JPY": JapanCalendar,
    "TOKYO": JapanCalendar,
}


_CALENDAR_CACHE: dict[str, Calendar] = {}


def get_calendar(name: str) -> Calendar:
    """
    Get a calendar by name (cached singletons).

    Args:
        name: Calendar name (e.g., "US", "TARGET", "UK", "JAPAN")

    Returns:
        Calendar instance

    Examples:
        >>> cal = get_calendar("US")
        >>> cal.is_business_day(date(2024, 7, 4))  # Independence Day
        False
    """
    key = name.upper().strip()

    if key in _CALENDAR_CACHE:
        return _CALENDAR_CACHE[key]

    # Check for combined calendars (e.g., "US+UK")
    if "+" in key:
        names = [n.strip() for n in key.split("+")]
        calendars = [get_calendar(n) for n in names]
        cal = CombinedCalendar(calendars)
        _CALENDAR_CACHE[key] = cal
        return cal

    if key not in _CALENDARS:
        valid = ", ".join(sorted(_CALENDARS.keys()))
        raise ValueError(f"Unknown calendar: {name}. Valid: {valid}")

    cal = _CALENDARS[key]()
    _CALENDAR_CACHE[key] = cal
    return cal
