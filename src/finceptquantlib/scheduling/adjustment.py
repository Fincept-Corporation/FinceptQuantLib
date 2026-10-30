"""
Business Day Adjustment Rules

Implements standard business day adjustment conventions:
- Following: Move to next business day
- Modified Following: Move to next, unless different month, then previous
- Preceding: Move to previous business day
- Modified Preceding: Move to previous, unless different month, then next
- Unadjusted: No adjustment
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, timedelta
from typing import Union

from finceptquantlib.scheduling.calendar import Calendar


class BusinessDayAdjustment(ABC):
    """
    Abstract base class for business day adjustments.

    Business day adjustments move non-business days to the nearest
    business day according to specific rules.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the adjustment name."""

    @abstractmethod
    def adjust(self, d: date, calendar: Calendar) -> date:
        """
        Adjust a date to a business day.

        Args:
            d: Date to adjust
            calendar: Calendar for business day determination

        Returns:
            Adjusted business day
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return self.name


class Following(BusinessDayAdjustment):
    """
    Following business day adjustment.

    If the date is not a business day, move to the next business day.

    Examples:
        >>> from finceptquantlib.scheduling.calendar import WeekendCalendar
        >>> adj = Following()
        >>> adj.adjust(date(2024, 1, 6), WeekendCalendar())  # Saturday -> Monday
        datetime.date(2024, 1, 8)
    """

    @property
    def name(self) -> str:
        return "Following"

    def adjust(self, d: date, calendar: Calendar) -> date:
        while not calendar.is_business_day(d):
            d += timedelta(days=1)
        return d


class ModifiedFollowing(BusinessDayAdjustment):
    """
    Modified Following business day adjustment.

    Move to the next business day, unless that would move to a different
    month, in which case move to the previous business day.

    This is the most common adjustment for money market and derivative
    instruments.

    Examples:
        >>> from finceptquantlib.scheduling.calendar import WeekendCalendar
        >>> adj = ModifiedFollowing()
        >>> # Jan 31, 2026 is Saturday
        >>> adj.adjust(date(2026, 1, 31), WeekendCalendar())  # Friday, not Feb 2
        datetime.date(2026, 1, 30)
    """

    @property
    def name(self) -> str:
        return "Modified Following"

    def adjust(self, d: date, calendar: Calendar) -> date:
        original_month = d.month
        adjusted = d

        # First try following
        while not calendar.is_business_day(adjusted):
            adjusted += timedelta(days=1)

        # If month changed, go backwards instead
        if adjusted.month != original_month:
            adjusted = d
            while not calendar.is_business_day(adjusted):
                adjusted -= timedelta(days=1)

        return adjusted


class Preceding(BusinessDayAdjustment):
    """
    Preceding business day adjustment.

    If the date is not a business day, move to the previous business day.

    Examples:
        >>> from finceptquantlib.scheduling.calendar import WeekendCalendar
        >>> adj = Preceding()
        >>> adj.adjust(date(2024, 1, 6), WeekendCalendar())  # Saturday -> Friday
        datetime.date(2024, 1, 5)
    """

    @property
    def name(self) -> str:
        return "Preceding"

    def adjust(self, d: date, calendar: Calendar) -> date:
        while not calendar.is_business_day(d):
            d -= timedelta(days=1)
        return d


class ModifiedPreceding(BusinessDayAdjustment):
    """
    Modified Preceding business day adjustment.

    Move to the previous business day, unless that would move to a different
    month, in which case move to the next business day.

    Examples:
        >>> from finceptquantlib.scheduling.calendar import WeekendCalendar
        >>> adj = ModifiedPreceding()
        >>> # Feb 1, 2026 is Sunday
        >>> adj.adjust(date(2026, 2, 1), WeekendCalendar())  # Monday, not Jan 30
        datetime.date(2026, 2, 2)
    """

    @property
    def name(self) -> str:
        return "Modified Preceding"

    def adjust(self, d: date, calendar: Calendar) -> date:
        original_month = d.month
        adjusted = d

        # First try preceding
        while not calendar.is_business_day(adjusted):
            adjusted -= timedelta(days=1)

        # If month changed, go forward instead
        if adjusted.month != original_month:
            adjusted = d
            while not calendar.is_business_day(adjusted):
                adjusted += timedelta(days=1)

        return adjusted


class Unadjusted(BusinessDayAdjustment):
    """
    No business day adjustment.

    Returns the date unchanged, even if it's not a business day.
    Used when accrual dates should not be adjusted.
    """

    @property
    def name(self) -> str:
        return "Unadjusted"

    def adjust(self, d: date, calendar: Calendar) -> date:
        return d


# Convenience function

def adjust_date(
    d: date,
    adjustment: Union[str, BusinessDayAdjustment] = "Modified Following",
    calendar: Union[str, Calendar] = "Weekend"
) -> date:
    """
    Adjust a date according to a business day convention.

    Args:
        d: Date to adjust
        adjustment: Adjustment rule (string or object)
        calendar: Calendar for business day determination

    Returns:
        Adjusted date

    Examples:
        >>> adjust_date(date(2024, 1, 6), "Following", "Weekend")
        datetime.date(2024, 1, 8)

        >>> adjust_date(date(2024, 7, 4), "Following", "US")  # Independence Day
        datetime.date(2024, 7, 5)
    """
    if isinstance(adjustment, str):
        adjustment = _get_adjustment(adjustment)

    if isinstance(calendar, str):
        from finceptquantlib.scheduling.calendar import get_calendar
        calendar = get_calendar(calendar)

    return adjustment.adjust(d, calendar)


_ADJUSTMENT_CACHE: dict[str, BusinessDayAdjustment] = {}

def _get_adjustment(name: str) -> BusinessDayAdjustment:
    """Get adjustment by name (cached)."""
    key = name.upper().strip().replace("_", " ").replace("-", " ")

    if key in _ADJUSTMENT_CACHE:
        return _ADJUSTMENT_CACHE[key]

    _ADJUSTMENTS_MAP = {
        "FOLLOWING": Following,
        "F": Following,
        "MODIFIED FOLLOWING": ModifiedFollowing,
        "MODFOLLOWING": ModifiedFollowing,
        "MF": ModifiedFollowing,
        "PRECEDING": Preceding,
        "P": Preceding,
        "MODIFIED PRECEDING": ModifiedPreceding,
        "MODPRECEDING": ModifiedPreceding,
        "MP": ModifiedPreceding,
        "UNADJUSTED": Unadjusted,
        "U": Unadjusted,
        "NONE": Unadjusted,
    }

    if key not in _ADJUSTMENTS_MAP:
        valid = ", ".join(sorted(set(_ADJUSTMENTS_MAP.keys())))
        raise ValueError(f"Unknown adjustment: {name}. Valid: {valid}")

    instance = _ADJUSTMENTS_MAP[key]()
    _ADJUSTMENT_CACHE[key] = instance
    return instance
