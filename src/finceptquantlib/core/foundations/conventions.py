"""
Standardized Conventions for FinceptQuantLib

This module defines library-wide conventions for:
- Date formats (ISO 8601: YYYY-MM-DD)
- Parameter naming standards
- Unit conventions
- Default values

All modules should import and use these conventions to ensure consistency.

Standard Date Format: ISO 8601 (YYYY-MM-DD)
------------------------------------------
The library uses ISO 8601 as the canonical date format:
- Internal storage: datetime.date objects
- String input: "YYYY-MM-DD" (primary), also accepts "YYYYMMDD", "DD/MM/YYYY"
- String output: Always "YYYY-MM-DD"
- Display format: "YYYY-MM-DD" (configurable via DateFormatter)

Standard Parameter Names:
------------------------
Dates:
- valuation_date: The date at which pricing/valuation is performed
- settlement_date: The date of cash settlement
- trade_date: The date the trade was executed
- effective_date: The start date of a contract/schedule
- termination_date: The end date of a contract/schedule
- maturity_date: The final maturity of an instrument
- issue_date: The date of issuance
- fixing_date: The date a rate is fixed/observed
- payment_date: The date of a payment
- accrual_start: Start of an accrual period
- accrual_end: End of an accrual period

Rates (always in decimal form, e.g., 0.05 = 5%):
- rate: Generic interest rate
- risk_free_rate: Risk-free interest rate
- dividend_yield: Dividend yield (continuous)
- forward_rate: Forward interest rate
- spot_rate: Spot/zero rate
- discount_rate: Discount rate
- coupon_rate: Bond coupon rate

Time (always in years as float):
- time_to_maturity: Time until maturity in years
- time_to_expiry: Time until expiry in years (for options)
- tenor: Period length in years
- holding_period: Holding period in years (or days with _days suffix)

Volatility (always in decimal form, e.g., 0.20 = 20%):
- volatility: Generic volatility (lognormal unless specified)
- sigma: Lognormal volatility
- normal_vol: Normal/Bachelier volatility
- local_vol: Local volatility

Prices/Values:
- spot: Current spot price
- strike: Strike price
- forward: Forward price
- notional: Notional/face value
- premium: Option premium
- present_value: Present value (PV)
- clean_price: Clean price (excluding accrued)
- dirty_price: Dirty price (including accrued)

Greeks (standard naming):
- delta: First derivative w.r.t. spot
- gamma: Second derivative w.r.t. spot
- theta: Time decay (per day)
- vega: Sensitivity to volatility (per 1% vol change)
- rho: Sensitivity to interest rate (per 1% rate change)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Union

from finceptquantlib.core.foundations.types import DateLike, to_date, RateUnit, VolatilityType


# =============================================================================
# Date Format Standards
# =============================================================================

# Standard ISO 8601 format
DATE_FORMAT_ISO = "%Y-%m-%d"
DATE_FORMAT_COMPACT = "%Y%m%d"
DATE_FORMAT_DISPLAY = "%Y-%m-%d"

# Accepted input formats (in order of preference)
DATE_INPUT_FORMATS = [
    "%Y-%m-%d",    # ISO 8601 (primary)
    "%Y%m%d",      # Compact ISO
    "%d/%m/%Y",    # European
    "%m/%d/%Y",    # US
    "%d-%m-%Y",    # European with dashes
    "%Y/%m/%d",    # Asian
]


class DateFormat(Enum):
    """Standard date format options."""
    ISO = "iso"           # YYYY-MM-DD (default)
    COMPACT = "compact"   # YYYYMMDD
    EUROPEAN = "european" # DD/MM/YYYY
    US = "us"             # MM/DD/YYYY

    def format_string(self) -> str:
        """Get the strftime format string."""
        formats = {
            DateFormat.ISO: "%Y-%m-%d",
            DateFormat.COMPACT: "%Y%m%d",
            DateFormat.EUROPEAN: "%d/%m/%Y",
            DateFormat.US: "%m/%d/%Y",
        }
        return formats[self]


@dataclass
class DateFormatter:
    """
    Configurable date formatter.

    The library default is ISO 8601 (YYYY-MM-DD).

    Example:
        >>> fmt = DateFormatter()  # Uses ISO by default
        >>> fmt.format_date(date(2024, 1, 15))
        '2024-01-15'

        >>> fmt = DateFormatter(DateFormat.EUROPEAN)
        >>> fmt.format_date(date(2024, 1, 15))
        '15/01/2024'
    """
    format: DateFormat = DateFormat.ISO

    def format_date(self, d: Union[date, datetime, DateLike]) -> str:
        """Format a date to string using the configured format."""
        if isinstance(d, datetime):
            d = d.date()
        elif not isinstance(d, date):
            d = to_date(d)
        return d.strftime(self.format.format_string())

    def parse(self, s: str) -> date:
        """Parse a date string (tries multiple formats)."""
        return to_date(s)


# Default formatter (ISO 8601)
DEFAULT_DATE_FORMATTER = DateFormatter(DateFormat.ISO)


def format_date(d: Union[date, datetime, DateLike], fmt: DateFormat = DateFormat.ISO) -> str:
    """
    Format a date to string.

    Args:
        d: Date to format
        fmt: Output format (default: ISO 8601)

    Returns:
        Formatted date string

    Example:
        >>> format_date(date(2024, 1, 15))
        '2024-01-15'
    """
    formatter = DateFormatter(fmt)
    return formatter.format_date(d)


def parse_date(s: str) -> date:
    """
    Parse a date string to date object.

    Accepts multiple formats, returns datetime.date.
    Primary format is ISO 8601 (YYYY-MM-DD).

    Args:
        s: Date string to parse

    Returns:
        date object

    Example:
        >>> parse_date("2024-01-15")
        datetime.date(2024, 1, 15)
        >>> parse_date("15/01/2024")  # European format
        datetime.date(2024, 1, 15)
    """
    return to_date(s)


# =============================================================================
# Rate Conventions
# =============================================================================

# RateUnit is imported from core.types (canonical location)


def normalize_rate(value: float, unit: RateUnit = RateUnit.DECIMAL) -> float:
    """
    Normalize rate to decimal form.

    Args:
        value: Rate value
        unit: Unit of input value

    Returns:
        Rate in decimal form (e.g., 0.05 for 5%)

    Example:
        >>> normalize_rate(5.0, RateUnit.PERCENT)
        0.05
        >>> normalize_rate(500, RateUnit.BASIS_POINTS)
        0.05
    """
    if unit == RateUnit.DECIMAL:
        return value
    elif unit == RateUnit.PERCENT:
        return value / 100.0
    elif unit == RateUnit.BASIS_POINTS:
        return value / 10000.0
    else:
        raise ValueError(f"Unknown rate unit: {unit}")


def rate_to_display(value: float, unit: RateUnit = RateUnit.PERCENT, decimals: int = 4) -> str:
    """
    Format rate for display.

    Args:
        value: Rate in decimal form
        unit: Desired display unit
        decimals: Number of decimal places

    Returns:
        Formatted string with unit suffix

    Example:
        >>> rate_to_display(0.05)
        '5.0000%'
        >>> rate_to_display(0.05, RateUnit.BASIS_POINTS)
        '500.0000 bps'
    """
    if unit == RateUnit.DECIMAL:
        return f"{value:.{decimals}f}"
    elif unit == RateUnit.PERCENT:
        return f"{value * 100:.{decimals}f}%"
    elif unit == RateUnit.BASIS_POINTS:
        return f"{value * 10000:.{decimals}f} bps"
    else:
        raise ValueError(f"Unknown rate unit: {unit}")


# =============================================================================
# Time Conventions
# =============================================================================

# Standard: Time is always in years
DAYS_PER_YEAR = 365.25  # Average including leap years
BUSINESS_DAYS_PER_YEAR = 252
MONTHS_PER_YEAR = 12


def days_to_years(days: int, convention: str = "actual") -> float:
    """
    Convert days to years.

    Args:
        days: Number of days
        convention: "actual" (365.25), "365" (365), "360", "business" (252)

    Returns:
        Time in years
    """
    if convention == "actual":
        return days / DAYS_PER_YEAR
    elif convention == "365":
        return days / 365.0
    elif convention == "360":
        return days / 360.0
    elif convention == "business":
        return days / BUSINESS_DAYS_PER_YEAR
    else:
        raise ValueError(f"Unknown convention: {convention}")


def years_to_days(years: float, convention: str = "actual") -> int:
    """
    Convert years to days.

    Args:
        years: Time in years
        convention: "actual" (365.25), "365" (365), "360", "business" (252)

    Returns:
        Number of days (rounded)
    """
    if convention == "actual":
        return int(round(years * DAYS_PER_YEAR))
    elif convention == "365":
        return int(round(years * 365.0))
    elif convention == "360":
        return int(round(years * 360.0))
    elif convention == "business":
        return int(round(years * BUSINESS_DAYS_PER_YEAR))
    else:
        raise ValueError(f"Unknown convention: {convention}")


# =============================================================================
# Volatility Conventions
# =============================================================================

# VolatilityType is imported from core.types (canonical location)


def normalize_volatility(
    value: float,
    vol_type: VolatilityType = VolatilityType.LOGNORMAL,
    is_percent: bool = False
) -> float:
    """
    Normalize volatility to decimal form.

    Args:
        value: Volatility value
        vol_type: Type of volatility (for documentation)
        is_percent: If True, divide by 100

    Returns:
        Volatility in decimal form
    """
    if is_percent:
        return value / 100.0
    return value


# =============================================================================
# Default Values
# =============================================================================

@dataclass(frozen=True)
class LibraryDefaults:
    """
    Default values used throughout the library.

    These can be overridden at the module or function level.
    """
    # Date
    date_format: DateFormat = DateFormat.ISO
    settlement_days: int = 2

    # Rates (in decimal)
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0

    # Time
    days_per_year: float = DAYS_PER_YEAR
    business_days_per_year: int = BUSINESS_DAYS_PER_YEAR

    # Volatility (in decimal)
    volatility: float = 0.20
    volatility_type: VolatilityType = VolatilityType.LOGNORMAL

    # Precision
    price_decimals: int = 6
    rate_decimals: int = 6
    greek_decimals: int = 6

    # Tolerance
    convergence_tolerance: float = 1e-10
    max_iterations: int = 100

    # Currency
    base_currency: str = "USD"


# Global defaults instance
DEFAULTS = LibraryDefaults()


# =============================================================================
# Parameter Name Aliases (for backward compatibility)
# =============================================================================

# Maps common alternative names to standard names
PARAMETER_ALIASES = {
    # Date parameters
    "as_of": "valuation_date",
    "as_of_date": "valuation_date",
    "pricing_date": "valuation_date",
    "value_date": "settlement_date",
    "settle_date": "settlement_date",
    "start_date": "effective_date",
    "end_date": "termination_date",
    "expiry_date": "maturity_date",
    "expiration_date": "maturity_date",

    # Rate parameters
    "r": "risk_free_rate",
    "rf": "risk_free_rate",
    "q": "dividend_yield",
    "div_yield": "dividend_yield",
    "y": "yield_rate",

    # Time parameters
    "T": "time_to_maturity",
    "t": "time_to_maturity",
    "tau": "time_to_maturity",
    "expiry": "time_to_expiry",
    "maturity": "time_to_maturity",

    # Volatility parameters
    "vol": "volatility",
    "sigma": "volatility",
    "sigma_n": "normal_vol",
    "sigma_normal": "normal_vol",

    # Price parameters
    "S": "spot",
    "S0": "spot",
    "K": "strike",
    "X": "strike",
    "F": "forward",
    "N": "notional",
    "face": "notional",
    "face_value": "notional",
    "principal": "notional",
}


def standardize_param_name(name: str) -> str:
    """
    Convert parameter name to standard form.

    Args:
        name: Parameter name (possibly non-standard)

    Returns:
        Standardized parameter name

    Example:
        >>> standardize_param_name("r")
        'risk_free_rate'
        >>> standardize_param_name("T")
        'time_to_maturity'
    """
    return PARAMETER_ALIASES.get(name, name)
