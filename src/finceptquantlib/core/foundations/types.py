"""
Enhanced Type System for FinceptQuantLib

Provides flexible, validated types for financial calculations:
- Money: Amount + currency with automatic conversion
- Rate: Handles bps, percentage, decimal automatically
- Tenor: Parse tenor strings like "1Y", "6M", "3M2D"
- DateLike: Accept multiple date formats

Also provides canonical enum definitions used across the library:
- OptionType: Call/Put
- BarrierType: Up/Down In/Out barrier types
- ExerciseType: European/American/Bermudan
- SwaptionType: Payer/Receiver
- Frequency: Payment frequencies
- VolatilityType: Lognormal/Normal quotation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum, auto, IntEnum
from typing import (
    Union, Optional, Tuple, List, Dict,
    Iterator
)


# =============================================================================
# Currency Definitions
# =============================================================================

class Currency(Enum):
    """ISO 4217 Currency codes."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CHF = "CHF"
    AUD = "AUD"
    CAD = "CAD"
    NZD = "NZD"
    SEK = "SEK"
    NOK = "NOK"
    DKK = "DKK"
    SGD = "SGD"
    HKD = "HKD"
    CNY = "CNY"
    CNH = "CNH"
    INR = "INR"
    KRW = "KRW"
    TWD = "TWD"
    THB = "THB"
    MYR = "MYR"
    IDR = "IDR"
    PHP = "PHP"
    VND = "VND"
    MXN = "MXN"
    BRL = "BRL"
    ARS = "ARS"
    CLP = "CLP"
    COP = "COP"
    PEN = "PEN"
    ZAR = "ZAR"
    TRY = "TRY"
    RUB = "RUB"
    PLN = "PLN"
    CZK = "CZK"
    HUF = "HUF"
    ILS = "ILS"
    AED = "AED"
    SAR = "SAR"
    QAR = "QAR"
    KWD = "KWD"
    BHD = "BHD"
    OMR = "OMR"
    EGP = "EGP"

    @classmethod
    def from_string(cls, s: str) -> Currency:
        """Parse currency from string."""
        s = s.upper().strip()
        try:
            return cls(s)
        except ValueError:
            raise ValueError(f"Unknown currency: {s}") from None


# Currency decimal places (for rounding)
CURRENCY_DECIMALS: Dict[Currency, int] = {
    Currency.JPY: 0,
    Currency.KRW: 0,
    Currency.VND: 0,
    Currency.IDR: 0,
    Currency.CLP: 0,
    Currency.KWD: 3,
    Currency.BHD: 3,
    Currency.OMR: 3,
}


def get_currency_decimals(ccy: Currency) -> int:
    """Get decimal places for a currency."""
    return CURRENCY_DECIMALS.get(ccy, 2)


# =============================================================================
# Option & Derivative Enums (Canonical Definitions)
# =============================================================================

class OptionType(str, Enum):
    """Option type: Call or Put. Parses "call"/"put" case-insensitively."""
    CALL = "call"
    PUT = "put"

    @classmethod
    def _missing_(cls, value: object) -> Optional["OptionType"]:
        if isinstance(value, str):
            return cls.__members__.get(value.strip().upper())
        return None


class BarrierType(str, Enum):
    """Types of single barrier options."""
    DOWN_AND_IN = "down_and_in"    # Becomes active when S touches barrier from above
    DOWN_AND_OUT = "down_and_out"  # Becomes worthless when S touches barrier from above
    UP_AND_IN = "up_and_in"        # Becomes active when S touches barrier from below
    UP_AND_OUT = "up_and_out"      # Becomes worthless when S touches barrier from below


class ExerciseType(Enum):
    """Option exercise style."""
    EUROPEAN = "european"   # Exercise only at expiry
    AMERICAN = "american"   # Exercise any time before expiry
    BERMUDAN = "bermudan"   # Exercise on specific dates


class SwaptionType(Enum):
    """Type of swaption."""
    PAYER = "payer"        # Right to enter payer swap (pay fixed)
    RECEIVER = "receiver"  # Right to enter receiver swap (receive fixed)


class VolatilityType(Enum):
    """Type of volatility quotation."""
    LOGNORMAL = "lognormal"  # Black-76 / Black-Scholes
    NORMAL = "normal"        # Bachelier


# =============================================================================
# Frequency Enum (Canonical Definition)
# =============================================================================

class Frequency(IntEnum):
    """Payment frequency enumeration."""
    ZERO_COUPON = 0
    ANNUAL = 1
    SEMI_ANNUAL = 2
    QUARTERLY = 4
    MONTHLY = 12
    WEEKLY = 52
    DAILY = 365

    @property
    def months(self) -> int:
        """Number of months per period."""
        if self == Frequency.ZERO_COUPON:
            return 0
        if self.value <= 12:
            return 12 // self.value
        return 0  # For WEEKLY/DAILY

    @property
    def periods_per_year(self) -> int:
        """Number of periods per year."""
        return self.value

    def __str__(self) -> str:
        names = {
            Frequency.ANNUAL: "Annual",
            Frequency.SEMI_ANNUAL: "Semi-Annual",
            Frequency.QUARTERLY: "Quarterly",
            Frequency.MONTHLY: "Monthly",
            Frequency.WEEKLY: "Weekly",
            Frequency.DAILY: "Daily",
            Frequency.ZERO_COUPON: "Zero Coupon",
        }
        return names.get(self, self.name)


class StubType(Enum):
    """Stub period type for schedule generation."""
    NONE = "none"
    SHORT_FRONT = "short_front"
    LONG_FRONT = "long_front"
    SHORT_BACK = "short_back"
    LONG_BACK = "long_back"


class RollConvention(Enum):
    """Roll date convention for schedule generation."""
    NONE = "none"
    EOM = "eom"            # End of month
    IMM = "imm"            # IMM dates (3rd Wednesday)
    SPECIFIC_DAY = "specific_day"  # Roll on specific day of month


# =============================================================================
# Rate Class
# =============================================================================

class RateUnit(Enum):
    """Standard rate units."""
    DECIMAL = "decimal"       # 0.05 = 5%
    PERCENT = "percent"       # 5 = 5%
    PERCENTAGE = "percent"    # Alias for PERCENT
    BASIS_POINTS = "bps"      # 500 = 5%


@dataclass(frozen=True, slots=True)
class Rate:
    """
    Flexible interest rate representation.

    Handles automatic conversion between decimal, percentage, and basis points.
    All internal storage is in decimal form.

    Examples:
        >>> Rate(0.05)              # 5% as decimal
        Rate(0.050000)
        >>> Rate("5%")              # 5% as string
        Rate(0.050000)
        >>> Rate(500, "bps")        # 500 basis points
        Rate(0.050000)
        >>> Rate.from_bps(50)       # 50 basis points
        Rate(0.005000)
        >>> Rate.from_percentage(5) # 5 percent
        Rate(0.050000)
    """
    _value: float  # Internal storage in decimal

    def __init__(
        self,
        value: Union[float, int, str, "Rate"],
        unit: Optional[str] = None
    ) -> None:
        """
        Create a Rate from various input formats.

        Args:
            value: Rate value (float, int, string like "5%", or another Rate)
            unit: Optional unit specification ("bps", "bp", "pct", "%", "decimal")
        """
        if isinstance(value, Rate):
            object.__setattr__(self, '_value', value._value)
            return

        if isinstance(value, str):
            parsed = self._parse_string(value)
            object.__setattr__(self, '_value', parsed)
            return

        # Numeric value with optional unit
        numeric = float(value)

        if unit is None:
            # Always treat as decimal — no ambiguous heuristic.
            # Use Rate.from_percentage() or Rate(value, "pct") for percentages.
            object.__setattr__(self, '_value', numeric)
        else:
            unit_lower = unit.lower().strip()
            if unit_lower in ("bps", "bp", "basis_points", "basis points"):
                object.__setattr__(self, '_value', numeric / 10000.0)
            elif unit_lower in ("pct", "%", "percent", "percentage"):
                object.__setattr__(self, '_value', numeric / 100.0)
            elif unit_lower in ("decimal", "dec"):
                object.__setattr__(self, '_value', numeric)
            else:
                raise ValueError(f"Unknown rate unit: {unit}")

    @staticmethod
    def _parse_string(s: str) -> float:
        """Parse rate from string format."""
        s = s.strip()

        # Handle percentage: "5%", "5.5%", "-2.5%"
        if s.endswith('%'):
            return float(s[:-1]) / 100.0

        # Handle basis points: "50bp", "50bps", "50 bps"
        bp_match = re.match(r'^(-?\d+\.?\d*)\s*(bps?|basis\s*points?)$', s, re.IGNORECASE)
        if bp_match:
            return float(bp_match.group(1)) / 10000.0

        # Handle plain decimal
        try:
            return float(s)
        except ValueError:
            raise ValueError(f"Cannot parse rate from string: {s}") from None

    @classmethod
    def from_bps(cls, bps: float) -> Rate:
        """Create Rate from basis points."""
        return cls(bps, "bps")

    @classmethod
    def from_percentage(cls, pct: float) -> Rate:
        """Create Rate from percentage."""
        return cls(pct, "pct")

    @classmethod
    def from_decimal(cls, dec: float) -> Rate:
        """Create Rate from decimal."""
        return cls(dec, "decimal")

    # Properties for different representations
    @property
    def decimal(self) -> float:
        """Rate as decimal (e.g., 0.05 for 5%)."""
        return self._value

    @property
    def value(self) -> float:
        """Alias for decimal."""
        return self._value

    @property
    def percentage(self) -> float:
        """Rate as percentage (e.g., 5.0 for 5%)."""
        return self._value * 100.0

    @property
    def pct(self) -> float:
        """Alias for percentage."""
        return self.percentage

    @property
    def bps(self) -> float:
        """Rate as basis points (e.g., 500 for 5%)."""
        return self._value * 10000.0

    @property
    def basis_points(self) -> float:
        """Alias for bps."""
        return self.bps

    # Arithmetic operations
    def __add__(self, other: Union[Rate, float]) -> Rate:
        if isinstance(other, Rate):
            return Rate(self._value + other._value)
        return Rate(self._value + float(other))

    def __radd__(self, other: float) -> Rate:
        return Rate(float(other) + self._value)

    def __sub__(self, other: Union[Rate, float]) -> Rate:
        if isinstance(other, Rate):
            return Rate(self._value - other._value)
        return Rate(self._value - float(other))

    def __rsub__(self, other: float) -> Rate:
        return Rate(float(other) - self._value)

    def __mul__(self, other: float) -> Rate:
        return Rate(self._value * float(other))

    def __rmul__(self, other: float) -> Rate:
        return Rate(float(other) * self._value)

    def __truediv__(self, other: float) -> Rate:
        return Rate(self._value / float(other))

    def __neg__(self) -> Rate:
        return Rate(-self._value)

    def __abs__(self) -> Rate:
        return Rate(abs(self._value))

    # Comparison operations
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Rate):
            return abs(self._value - other._value) < 1e-12
        if isinstance(other, (int, float)):
            return abs(self._value - float(other)) < 1e-12
        return NotImplemented

    def __lt__(self, other: Union[Rate, float]) -> bool:
        if isinstance(other, Rate):
            return self._value < other._value
        return self._value < float(other)

    def __le__(self, other: Union[Rate, float]) -> bool:
        if isinstance(other, Rate):
            return self._value <= other._value
        return self._value <= float(other)

    def __gt__(self, other: Union[Rate, float]) -> bool:
        if isinstance(other, Rate):
            return self._value > other._value
        return self._value > float(other)

    def __ge__(self, other: Union[Rate, float]) -> bool:
        if isinstance(other, Rate):
            return self._value >= other._value
        return self._value >= float(other)

    def __hash__(self) -> int:
        return hash(self._value)

    def __float__(self) -> float:
        return self._value

    def __repr__(self) -> str:
        return f"Rate({self._value:.6f})"

    def __str__(self) -> str:
        if abs(self._value) < 0.0001:
            return f"{self.bps:.2f}bps"
        return f"{self.percentage:.4f}%"

    def format(self, unit: str = "pct", decimals: int = 4) -> str:
        """Format rate with specified unit and precision."""
        unit_lower = unit.lower()
        if unit_lower in ("bps", "bp"):
            return f"{self.bps:.{decimals}f}bps"
        elif unit_lower in ("pct", "%"):
            return f"{self.percentage:.{decimals}f}%"
        else:
            return f"{self._value:.{decimals}f}"


# =============================================================================
# Money Class
# =============================================================================

@dataclass(slots=True)
class Money:
    """
    Money amount with currency.

    Supports arithmetic operations and currency conversion.

    Examples:
        >>> Money(1_000_000, "USD")
        Money(1,000,000.00, 'USD')
        >>> Money(1_000_000, Currency.EUR)
        Money(1,000,000.00, 'EUR')
        >>> usd + eur  # Raises error - use convert first  # doctest: +SKIP
        >>> usd.convert("EUR", fx_rate=1.08)  # doctest: +SKIP
    """
    amount: float
    currency: Currency

    def __init__(
        self,
        amount: Union[float, int, Decimal],
        currency: Union[str, Currency]
    ) -> None:
        self.amount = float(amount)
        if isinstance(currency, str):
            self.currency = Currency.from_string(currency)
        else:
            self.currency = currency

    @property
    def ccy(self) -> str:
        """Currency code as string."""
        return self.currency.value

    def round(self, decimals: Optional[int] = None) -> Money:
        """Round to currency-appropriate decimals."""
        if decimals is None:
            decimals = get_currency_decimals(self.currency)
        return Money(round(self.amount, decimals), self.currency)

    def convert(
        self,
        target: Union[str, Currency],
        fx_rate: float,
        invert: bool = False
    ) -> Money:
        """
        Convert to another currency.

        Args:
            target: Target currency
            fx_rate: Exchange rate (units of target per unit of source)
            invert: If True, fx_rate is source per target (will be inverted)

        Returns:
            Money in target currency
        """
        if isinstance(target, str):
            target = Currency.from_string(target)

        rate = 1.0 / fx_rate if invert else fx_rate
        return Money(self.amount * rate, target)

    def __add__(self, other: Money) -> Money:
        if not isinstance(other, Money):
            return NotImplemented
        if self.currency != other.currency:
            raise ValueError(
                f"Cannot add {self.currency.value} and {other.currency.value}. "
                "Convert to same currency first."
            )
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: Money) -> Money:
        if not isinstance(other, Money):
            return NotImplemented
        if self.currency != other.currency:
            raise ValueError(
                f"Cannot subtract {self.currency.value} and {other.currency.value}. "
                "Convert to same currency first."
            )
        return Money(self.amount - other.amount, self.currency)

    def __mul__(self, other: float) -> Money:
        return Money(self.amount * float(other), self.currency)

    def __rmul__(self, other: float) -> Money:
        return Money(float(other) * self.amount, self.currency)

    def __truediv__(self, other: float) -> Money:
        return Money(self.amount / float(other), self.currency)

    def __neg__(self) -> Money:
        return Money(-self.amount, self.currency)

    def __abs__(self) -> Money:
        return Money(abs(self.amount), self.currency)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Money):
            return (
                self.currency == other.currency and
                abs(self.amount - other.amount) < 1e-10
            )
        return NotImplemented

    def __lt__(self, other: Money) -> bool:
        if self.currency != other.currency:
            raise ValueError("Cannot compare different currencies")
        return self.amount < other.amount

    def __le__(self, other: Money) -> bool:
        if self.currency != other.currency:
            raise ValueError("Cannot compare different currencies")
        return self.amount <= other.amount

    def __repr__(self) -> str:
        return f"Money({self.amount:,.2f}, {self.currency.value!r})"

    def __str__(self) -> str:
        decimals = get_currency_decimals(self.currency)
        return f"{self.currency.value} {self.amount:,.{decimals}f}"

    def format(self, show_symbol: bool = True, decimals: Optional[int] = None) -> str:
        """Format money with optional symbol."""
        if decimals is None:
            decimals = get_currency_decimals(self.currency)
        if show_symbol:
            return f"{self.currency.value} {self.amount:,.{decimals}f}"
        return f"{self.amount:,.{decimals}f}"


# =============================================================================
# Tenor Class
# =============================================================================

@dataclass(frozen=True, slots=True)
class Tenor:
    """
    Tenor/period representation.

    Parses and stores tenors like "1Y", "6M", "3M2D", "O/N", "1W".

    Examples:
        >>> Tenor("1Y")
        Tenor('1Y')
        >>> Tenor("6M")
        Tenor('6M')
        >>> Tenor("1Y6M")
        Tenor('1Y6M')
        >>> Tenor("3M2D")
        Tenor('3M2D')
        >>> Tenor("O/N")  # Overnight
        Tenor('O/N')
        >>> Tenor("T/N")  # Tomorrow/Next
        Tenor('T/N')
        >>> Tenor("1W")   # 1 Week
        Tenor('1W')
        >>> Tenor(years=1, months=6)
        Tenor('1Y6M')
    """
    years: int = 0
    months: int = 0
    weeks: int = 0
    days: int = 0
    is_overnight: bool = False
    is_tom_next: bool = False
    is_spot_next: bool = False

    _TENOR_PATTERN = re.compile(
        r'^(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)W)?(?:(\d+)D)?$',
        re.IGNORECASE
    )

    def __init__(
        self,
        tenor: Optional[str] = None,
        *,
        years: int = 0,
        months: int = 0,
        weeks: int = 0,
        days: int = 0
    ) -> None:
        if tenor is not None:
            parsed = self._parse(tenor)
            object.__setattr__(self, 'years', parsed.years)
            object.__setattr__(self, 'months', parsed.months)
            object.__setattr__(self, 'weeks', parsed.weeks)
            object.__setattr__(self, 'days', parsed.days)
            object.__setattr__(self, 'is_overnight', parsed.is_overnight)
            object.__setattr__(self, 'is_tom_next', parsed.is_tom_next)
            object.__setattr__(self, 'is_spot_next', parsed.is_spot_next)
        else:
            object.__setattr__(self, 'years', years)
            object.__setattr__(self, 'months', months)
            object.__setattr__(self, 'weeks', weeks)
            object.__setattr__(self, 'days', days)
            object.__setattr__(self, 'is_overnight', False)
            object.__setattr__(self, 'is_tom_next', False)
            object.__setattr__(self, 'is_spot_next', False)

    @classmethod
    def _parse(cls, s: str) -> Tenor:
        """Parse tenor string."""
        s = s.upper().strip()

        # Special tenors
        if s in ("O/N", "ON", "OVERNIGHT"):
            t = object.__new__(cls)
            object.__setattr__(t, 'years', 0)
            object.__setattr__(t, 'months', 0)
            object.__setattr__(t, 'weeks', 0)
            object.__setattr__(t, 'days', 1)
            object.__setattr__(t, 'is_overnight', True)
            object.__setattr__(t, 'is_tom_next', False)
            object.__setattr__(t, 'is_spot_next', False)
            return t

        if s in ("T/N", "TN", "TOM/NEXT", "TOMNEXT"):
            t = object.__new__(cls)
            object.__setattr__(t, 'years', 0)
            object.__setattr__(t, 'months', 0)
            object.__setattr__(t, 'weeks', 0)
            object.__setattr__(t, 'days', 1)
            object.__setattr__(t, 'is_overnight', False)
            object.__setattr__(t, 'is_tom_next', True)
            object.__setattr__(t, 'is_spot_next', False)
            return t

        if s in ("S/N", "SN", "SPOT/NEXT", "SPOTNEXT"):
            t = object.__new__(cls)
            object.__setattr__(t, 'years', 0)
            object.__setattr__(t, 'months', 0)
            object.__setattr__(t, 'weeks', 0)
            object.__setattr__(t, 'days', 1)
            object.__setattr__(t, 'is_overnight', False)
            object.__setattr__(t, 'is_tom_next', False)
            object.__setattr__(t, 'is_spot_next', True)
            return t

        # Standard tenor pattern
        match = cls._TENOR_PATTERN.match(s)
        if match:
            years = int(match.group(1) or 0)
            months = int(match.group(2) or 0)
            weeks = int(match.group(3) or 0)
            days = int(match.group(4) or 0)

            t = object.__new__(cls)
            object.__setattr__(t, 'years', years)
            object.__setattr__(t, 'months', months)
            object.__setattr__(t, 'weeks', weeks)
            object.__setattr__(t, 'days', days)
            object.__setattr__(t, 'is_overnight', False)
            object.__setattr__(t, 'is_tom_next', False)
            object.__setattr__(t, 'is_spot_next', False)
            return t

        # Try parsing as just a number of months (e.g., "18" -> 18M)
        if s.isdigit():
            months = int(s)
            t = object.__new__(cls)
            object.__setattr__(t, 'years', 0)
            object.__setattr__(t, 'months', months)
            object.__setattr__(t, 'weeks', 0)
            object.__setattr__(t, 'days', 0)
            object.__setattr__(t, 'is_overnight', False)
            object.__setattr__(t, 'is_tom_next', False)
            object.__setattr__(t, 'is_spot_next', False)
            return t

        raise ValueError(f"Cannot parse tenor: {s}")

    @property
    def total_months(self) -> int:
        """Total tenor in months (approximate for weeks/days)."""
        return self.years * 12 + self.months

    @property
    def total_days(self) -> int:
        """Approximate total days (using 30-day months, 365-day years)."""
        return self.years * 365 + self.months * 30 + self.weeks * 7 + self.days

    @property
    def year_fraction(self) -> float:
        """Approximate year fraction."""
        return self.total_days / 365.0

    def add_to_date(self, dt: date) -> date:
        """Add this tenor to a date."""
        # Add years and months
        year = dt.year + self.years
        month = dt.month + self.months

        while month > 12:
            month -= 12
            year += 1
        while month < 1:
            month += 12
            year -= 1

        # Handle end-of-month
        day = min(dt.day, _days_in_month(year, month))
        result = date(year, month, day)

        # Add weeks and days
        result += timedelta(weeks=self.weeks, days=self.days)

        return result

    def __add__(self, other: Tenor) -> Tenor:
        return Tenor(
            years=self.years + other.years,
            months=self.months + other.months,
            weeks=self.weeks + other.weeks,
            days=self.days + other.days
        )

    def __mul__(self, n: int) -> Tenor:
        return Tenor(
            years=self.years * n,
            months=self.months * n,
            weeks=self.weeks * n,
            days=self.days * n
        )

    def __rmul__(self, n: int) -> Tenor:
        return self.__mul__(n)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Tenor):
            return (
                self.years == other.years and
                self.months == other.months and
                self.weeks == other.weeks and
                self.days == other.days and
                self.is_overnight == other.is_overnight and
                self.is_tom_next == other.is_tom_next and
                self.is_spot_next == other.is_spot_next
            )
        if isinstance(other, str):
            return self == Tenor(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.years, self.months, self.weeks, self.days,
                     self.is_overnight, self.is_tom_next, self.is_spot_next))

    def __repr__(self) -> str:
        if self.is_overnight:
            return "Tenor('O/N')"
        if self.is_tom_next:
            return "Tenor('T/N')"
        if self.is_spot_next:
            return "Tenor('S/N')"
        return f"Tenor('{self}')"

    def __str__(self) -> str:
        if self.is_overnight:
            return "O/N"
        if self.is_tom_next:
            return "T/N"
        if self.is_spot_next:
            return "S/N"

        parts = []
        if self.years:
            parts.append(f"{self.years}Y")
        if self.months:
            parts.append(f"{self.months}M")
        if self.weeks:
            parts.append(f"{self.weeks}W")
        if self.days:
            parts.append(f"{self.days}D")

        return "".join(parts) if parts else "0D"


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


# =============================================================================
# DateLike Protocol and Utilities
# =============================================================================

# Type alias for date-like inputs
DateLike = Union[date, datetime, str, int]


def to_date(value: DateLike) -> date:
    """
    Convert various date formats to date object.

    Accepts:
        - date object (returned as-is)
        - datetime object (date extracted)
        - string in ISO format "YYYY-MM-DD" or "YYYYMMDD"
        - int as Excel serial date number

    Args:
        value: Date-like value to convert

    Returns:
        date object
    """
    if isinstance(value, date) and not isinstance(value, datetime):
        return value

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, str):
        s = value.strip()
        # Try ISO format: YYYY-MM-DD
        if len(s) == 10 and s[4] == '-' and s[7] == '-':
            return date.fromisoformat(s)
        # Try compact format: YYYYMMDD
        if len(s) == 8 and s.isdigit():
            return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
        # Try other common formats
        for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y"):
            try:
                return datetime.strptime(s, fmt).date()
            except ValueError:
                continue
        raise ValueError(f"Cannot parse date from string: {value}")

    if isinstance(value, int):
        # Excel serial date (days since 1899-12-30)
        # Note: Excel incorrectly treats 1900 as leap year
        if value < 1:
            raise ValueError(f"Invalid Excel date serial: {value}")

        # Excel epoch is 1899-12-30 (day 0)
        # But Excel has a bug: it thinks 1900-02-29 exists
        # So we need to adjust for dates after Feb 28, 1900
        if value >= 60:
            value -= 1  # Adjust for Excel's leap year bug

        epoch = date(1899, 12, 30)
        return epoch + timedelta(days=value)

    raise TypeError(f"Cannot convert {type(value).__name__} to date")


def to_datetime(value: DateLike) -> datetime:
    """Convert various formats to datetime object."""
    if isinstance(value, datetime):
        return value
    return datetime.combine(to_date(value), datetime.min.time())


def date_to_excel(dt: date) -> int:
    """Convert date to Excel serial number."""
    epoch = date(1899, 12, 30)
    days = (dt - epoch).days
    if days >= 60:
        days += 1  # Adjust for Excel's leap year bug
    return days


# =============================================================================
# Spread Types
# =============================================================================

@dataclass(frozen=True, slots=True)
class Spread:
    """
    Credit or basis spread representation.

    Similar to Rate but semantically represents a spread over a reference rate.
    """
    _value: float  # Internal storage in decimal

    def __init__(self, value: Union[float, int, str], unit: Optional[str] = None) -> None:
        if isinstance(value, str):
            # Reuse Rate's parsing
            rate = Rate(value)
            object.__setattr__(self, '_value', rate._value)
        elif unit is not None:
            rate = Rate(value, unit)
            object.__setattr__(self, '_value', rate._value)
        else:
            object.__setattr__(self, '_value', float(value))

    @classmethod
    def from_bps(cls, bps: float) -> Spread:
        return cls(bps, "bps")

    @property
    def decimal(self) -> float:
        return self._value

    @property
    def bps(self) -> float:
        return self._value * 10000.0

    @property
    def percentage(self) -> float:
        return self._value * 100.0

    def __add__(self, other: Union[Spread, Rate, float]) -> Spread:
        if isinstance(other, (Spread, Rate)):
            return Spread(self._value + other._value)
        return Spread(self._value + float(other))

    def __radd__(self, other: Union[Rate, float]) -> Rate:
        """Adding spread to rate returns a rate."""
        if isinstance(other, Rate):
            return Rate(other._value + self._value)
        return Rate(float(other) + self._value)

    def __float__(self) -> float:
        return self._value

    def __repr__(self) -> str:
        return f"Spread({self.bps:.2f}bps)"

    def __str__(self) -> str:
        return f"{self.bps:.2f}bps"


# =============================================================================
# Notional Schedule
# =============================================================================

class NotionalType(Enum):
    """Type of notional schedule."""
    CONSTANT = auto()
    AMORTIZING = auto()
    ACCRETING = auto()
    ROLLER_COASTER = auto()
    CUSTOM = auto()


@dataclass(frozen=True, slots=True)
class NotionalStep:
    """Single notional schedule step."""
    date: date
    notional: float


@dataclass
class NotionalSchedule:
    """
    Flexible notional schedule for amortizing/accreting/custom notional patterns.

    Supports both index-based access (for core legs) and date-based access
    (for advanced swaps).

    Examples:
        >>> NotionalSchedule.constant(1_000_000)
        NotionalSchedule(steps=[], initial_notional=1000000, _notional_type=<NotionalType.CONSTANT: 1>)
        >>> NotionalSchedule.linear_amortization(1_000_000, periods=10)
        NotionalSchedule(steps=[], initial_notional=1000000, _notional_type=<NotionalType.AMORTIZING: 2>)
        >>> NotionalSchedule.custom([1_000_000, 900_000, 800_000])
        NotionalSchedule(steps=[], initial_notional=1000000, _notional_type=<NotionalType.CUSTOM: 5>)
    """
    steps: List[NotionalStep] = field(default_factory=list)
    initial_notional: float = 0.0
    _notionals: List[float] = field(default_factory=list, repr=False)
    _notional_type: NotionalType = NotionalType.CONSTANT

    @classmethod
    def constant(cls, notional: float) -> "NotionalSchedule":
        """Create constant notional schedule."""
        return cls(
            steps=[],
            initial_notional=notional,
            _notionals=[notional],
            _notional_type=NotionalType.CONSTANT,
        )

    @classmethod
    def linear_amortization(
        cls,
        initial: float,
        periods: int,
        final: float = 0.0
    ) -> "NotionalSchedule":
        """Linear amortization from initial to final over periods."""
        step = (initial - final) / periods
        notionals = [initial - step * i for i in range(periods + 1)]
        return cls(
            steps=[],
            initial_notional=initial,
            _notionals=notionals,
            _notional_type=NotionalType.AMORTIZING,
        )

    @classmethod
    def linear_amortizing(
        cls,
        initial: float,
        final: float,
        start_date: date,
        end_date: date,
        frequency: "Frequency" = None,
    ) -> "NotionalSchedule":
        """
        Create linearly amortizing schedule with date-based steps.

        Args:
            initial: Starting notional
            final: Ending notional (typically 0)
            start_date: First amortization date
            end_date: Last date
            frequency: Amortization frequency
        """
        from finceptquantlib.scheduling.schedule import Schedule, Frequency as SchedFreq
        if frequency is None:
            frequency = SchedFreq.ANNUAL

        schedule = Schedule(
            effective_date=start_date,
            termination_date=end_date,
            frequency=frequency,
        )

        periods = schedule.periods
        num_steps = len(periods)
        steps = []

        if num_steps > 0:
            step_amount = (initial - final) / num_steps
            current = initial
            for _, period_end in periods:
                current -= step_amount
                steps.append(NotionalStep(date=period_end, notional=max(0, current)))

        notionals = [initial] + [s.notional for s in steps]
        return cls(
            steps=steps,
            initial_notional=initial,
            _notionals=notionals,
            _notional_type=NotionalType.AMORTIZING,
        )

    @classmethod
    def bullet_amortizing(
        cls,
        notional: float,
        amort_dates: List[tuple],
    ) -> "NotionalSchedule":
        """
        Create schedule with specific amortization events.

        Args:
            notional: Initial notional
            amort_dates: List of (date, amortization_amount) tuples
        """
        steps = []
        remaining = notional

        for amort_date, amount in sorted(amort_dates, key=lambda x: x[0]):
            remaining -= amount
            steps.append(NotionalStep(date=amort_date, notional=max(0, remaining)))

        notionals = [notional] + [s.notional for s in steps]
        return cls(
            steps=steps,
            initial_notional=notional,
            _notionals=notionals,
            _notional_type=NotionalType.AMORTIZING,
        )

    @classmethod
    def mortgage_style(
        cls,
        initial: float,
        rate: float,
        start_date: date,
        end_date: date,
        frequency: "Frequency" = None,
    ) -> "NotionalSchedule":
        """
        Create mortgage-style amortization (constant payment).

        Args:
            initial: Initial principal
            rate: Annual interest rate
            start_date: Start date
            end_date: End date
            frequency: Payment frequency
        """
        from finceptquantlib.scheduling.schedule import Schedule, Frequency as SchedFreq
        if frequency is None:
            frequency = SchedFreq.MONTHLY

        schedule = Schedule(
            effective_date=start_date,
            termination_date=end_date,
            frequency=frequency,
        )

        periods = schedule.periods
        n_payments = len(periods)

        if n_payments == 0:
            return cls.constant(initial)

        periods_per_year = {
            SchedFreq.MONTHLY: 12,
            SchedFreq.QUARTERLY: 4,
            SchedFreq.SEMI_ANNUAL: 2,
            SchedFreq.ANNUAL: 1,
        }.get(frequency, 12)

        periodic_rate = rate / periods_per_year

        if periodic_rate > 0:
            payment = initial * periodic_rate / (1 - (1 + periodic_rate) ** -n_payments)
        else:
            payment = initial / n_payments

        steps = []
        balance = initial

        for _, period_end in periods:
            interest = balance * periodic_rate
            principal = payment - interest
            balance -= principal
            steps.append(NotionalStep(date=period_end, notional=max(0, balance)))

        notionals = [initial] + [s.notional for s in steps]
        return cls(
            steps=steps,
            initial_notional=initial,
            _notionals=notionals,
            _notional_type=NotionalType.AMORTIZING,
        )

    @classmethod
    def custom(cls, notionals: List[float]) -> "NotionalSchedule":
        """Custom notional schedule from a list of values."""
        return cls(
            steps=[],
            initial_notional=notionals[0] if notionals else 0.0,
            _notionals=list(notionals),
            _notional_type=NotionalType.CUSTOM,
        )

    def get_notional(self, period_index_or_date=None) -> float:
        """
        Get notional for a specific period index or date.

        Args:
            period_index_or_date: int (period index) or date object
        """
        if period_index_or_date is None:
            return self.initial_notional

        # Date-based lookup
        if isinstance(period_index_or_date, date):
            as_of = period_index_or_date
            notional = self.initial_notional
            for step in self.steps:
                if step.date <= as_of:
                    notional = step.notional
                else:
                    break
            return notional

        # Index-based lookup
        period_index = int(period_index_or_date)
        if self._notional_type == NotionalType.CONSTANT:
            return self.initial_notional
        if period_index >= len(self._notionals):
            return self._notionals[-1] if self._notionals else self.initial_notional
        return self._notionals[period_index]

    def get_average_notional(self, start: date, end: date) -> float:
        """Get average notional over a period (for accrued interest)."""
        mid = start + (end - start) / 2
        return self.get_notional(mid)

    def __len__(self) -> int:
        if self._notionals:
            return len(self._notionals)
        return len(self.steps) + 1

    def __iter__(self) -> Iterator[float]:
        if self._notionals:
            return iter(self._notionals)
        return iter([self.initial_notional] + [s.notional for s in self.steps])


# =============================================================================
# Input Validation Helpers
# =============================================================================

def validate_positive(value: float, name: str = "value") -> float:
    """Validate that a value is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def validate_non_negative(value: float, name: str = "value") -> float:
    """Validate that a value is non-negative."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def validate_in_range(
    value: float,
    min_val: float,
    max_val: float,
    name: str = "value"
) -> float:
    """Validate that a value is within a range."""
    if not (min_val <= value <= max_val):
        raise ValueError(f"{name} must be in [{min_val}, {max_val}], got {value}")
    return value


def validate_date_order(
    start: date,
    end: date,
    start_name: str = "start",
    end_name: str = "end"
) -> Tuple[date, date]:
    """Validate that start date is before end date."""
    if start >= end:
        raise ValueError(f"{start_name} ({start}) must be before {end_name} ({end})")
    return start, end


# =============================================================================
# Type Coercion Helpers
# =============================================================================

def as_rate(value: Union[Rate, float, str]) -> Rate:
    """Coerce value to Rate."""
    if isinstance(value, Rate):
        return value
    return Rate(value)


def as_date(value: DateLike) -> date:
    """Coerce value to date."""
    return to_date(value)


def as_tenor(value: Union[Tenor, str]) -> Tenor:
    """Coerce value to Tenor."""
    if isinstance(value, Tenor):
        return value
    return Tenor(value)


def as_currency(value: Union[Currency, str]) -> Currency:
    """Coerce value to Currency."""
    if isinstance(value, Currency):
        return value
    return Currency.from_string(value)


def as_money(
    amount: Union[Money, float, int],
    currency: Optional[Union[Currency, str]] = None
) -> Money:
    """Coerce value to Money."""
    if isinstance(amount, Money):
        return amount
    if currency is None:
        raise ValueError("Currency required when amount is not Money")
    return Money(amount, currency)
