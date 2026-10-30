"""
Calendar-Based Features

Date-based feature generators for financial machine learning.
Extract temporal patterns such as month-of-year, day-of-week,
quarter boundaries, and trading day counts.

All functions accept List[date] and return List[int].
Uses only the standard library ``datetime`` module.

Functions
---------
- month_of_year, quarter, day_of_week, day_of_month
- is_month_end, is_quarter_end, is_year_end
- days_to_expiry, trading_day_of_month
"""

from __future__ import annotations

import calendar
from datetime import date
from typing import List


def month_of_year(dates: List[date]) -> List[int]:
    """Extract month of year from dates.

    Parameters
    ----------
    dates : List[date]
        List of dates.

    Returns
    -------
    List[int]
        Month numbers, 1 (January) through 12 (December).
    """
    return [d.month for d in dates]


def quarter(dates: List[date]) -> List[int]:
    """Extract fiscal quarter from dates.

    Parameters
    ----------
    dates : List[date]
        List of dates.

    Returns
    -------
    List[int]
        Quarter numbers, 1 through 4.
    """
    return [(d.month - 1) // 3 + 1 for d in dates]


def day_of_week(dates: List[date]) -> List[int]:
    """Extract day of week from dates.

    Parameters
    ----------
    dates : List[date]
        List of dates.

    Returns
    -------
    List[int]
        Day of week: 0 = Monday through 4 = Friday, 5 = Saturday,
        6 = Sunday.
    """
    return [d.weekday() for d in dates]


def day_of_month(dates: List[date]) -> List[int]:
    """Extract day of month from dates.

    Parameters
    ----------
    dates : List[date]
        List of dates.

    Returns
    -------
    List[int]
        Day of month, 1 through 31.
    """
    return [d.day for d in dates]


def is_month_end(dates: List[date]) -> List[int]:
    """Check if each date is the last day of its month.

    Parameters
    ----------
    dates : List[date]
        List of dates.

    Returns
    -------
    List[int]
        1 if month end, 0 otherwise.
    """
    result: List[int] = []
    for d in dates:
        last_day = calendar.monthrange(d.year, d.month)[1]
        result.append(1 if d.day == last_day else 0)
    return result


def is_quarter_end(dates: List[date]) -> List[int]:
    """Check if each date is the last day of a calendar quarter.

    Parameters
    ----------
    dates : List[date]
        List of dates.

    Returns
    -------
    List[int]
        1 if quarter end (Mar 31, Jun 30, Sep 30, Dec 31), 0 otherwise.
    """
    quarter_end_months = {3, 6, 9, 12}
    result: List[int] = []
    for d in dates:
        last_day = calendar.monthrange(d.year, d.month)[1]
        if d.month in quarter_end_months and d.day == last_day:
            result.append(1)
        else:
            result.append(0)
    return result


def is_year_end(dates: List[date]) -> List[int]:
    """Check if each date is December 31.

    Parameters
    ----------
    dates : List[date]
        List of dates.

    Returns
    -------
    List[int]
        1 if December 31, 0 otherwise.
    """
    return [1 if d.month == 12 and d.day == 31 else 0 for d in dates]


def days_to_expiry(dates: List[date], expiry: date) -> List[int]:
    """Compute calendar days remaining until an expiry date.

    Parameters
    ----------
    dates : List[date]
        List of observation dates.
    expiry : date
        Expiry or maturity date.

    Returns
    -------
    List[int]
        Number of calendar days from each date to expiry.
        Negative values indicate the date is past expiry.
    """
    return [(expiry - d).days for d in dates]


def trading_day_of_month(dates: List[date]) -> List[int]:
    """Approximate trading day of month (excludes weekends).

    Parameters
    ----------
    dates : List[date]
        List of dates.

    Returns
    -------
    List[int]
        Trading day number within the month, starting at 1.
        Counts only weekdays (Monday-Friday) from the first of
        the month up to and including the given date.

    Notes
    -----
    This is an approximation that excludes weekends but does not
    account for public holidays.
    """
    result: List[int] = []
    for d in dates:
        count = 0
        for day_num in range(1, d.day + 1):
            check = date(d.year, d.month, day_num)
            if check.weekday() < 5:  # Monday=0 through Friday=4
                count += 1
        result.append(count)
    return result
