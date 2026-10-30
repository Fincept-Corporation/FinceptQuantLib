"""
Scheduling module for date handling, calendars, and day count conventions.

This module provides:
- Day count conventions (ACT/360, ACT/365, 30/360, etc.)
- Business day calendars for major financial centers
- Date adjustment rules (following, modified following, etc.)
- Payment schedule generation
"""

from finceptquantlib.scheduling.daycount import (
    DayCountConvention,
    Actual360,
    Actual365Fixed,
    Actual365Actual,
    ActualActualISDA,
    Thirty360,
    Thirty360European,
    ThirtyE360,
    ThirtyE360ISDA,
    Business252,
    year_fraction,
)

from finceptquantlib.scheduling.calendar import (
    Calendar,
    WeekendCalendar,
    TargetCalendar,
    USFederalCalendar,
    UKCalendar,
    JapanCalendar,
    CombinedCalendar,
    get_calendar,
)

from finceptquantlib.scheduling.adjustment import (
    BusinessDayAdjustment,
    adjust_date,
    Following,
    ModifiedFollowing,
    Preceding,
    ModifiedPreceding,
    Unadjusted,
)

from finceptquantlib.scheduling.schedule import (
    Schedule,
    Frequency,
    StubType,
    generate_schedule,
    RollConvention,
    SchedulePeriod,
)

__all__ = [
    # Day count conventions
    "DayCountConvention",
    "Actual360",
    "Actual365Fixed",
    "Actual365Actual",
    "ActualActualISDA",
    "Thirty360",
    "Thirty360European",
    "ThirtyE360",
    "ThirtyE360ISDA",
    "Business252",
    "year_fraction",
    # Calendars
    "Calendar",
    "WeekendCalendar",
    "TargetCalendar",
    "USFederalCalendar",
    "UKCalendar",
    "JapanCalendar",
    "CombinedCalendar",
    "get_calendar",
    # Adjustments
    "BusinessDayAdjustment",
    "adjust_date",
    "Following",
    "ModifiedFollowing",
    "Preceding",
    "ModifiedPreceding",
    "Unadjusted",
    # Schedule
    "Schedule",
    "Frequency",
    "StubType",
    "generate_schedule",
    "RollConvention",
    "SchedulePeriod",
]
