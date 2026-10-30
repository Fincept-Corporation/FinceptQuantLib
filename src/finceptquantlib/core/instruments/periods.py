"""
Period Module - Atomic Payment Units

Provides granular period types that represent individual payments:
- FixedCouponPeriod: Single fixed rate payment
- FloatCouponPeriod: Single floating rate payment with fixing
- CompoundedRFRPeriod: RFR compounding (SOFR, SONIA, etc.)
- InflationPeriod: Inflation-linked payment
- PrincipalExchange: Principal payment
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum, auto
from typing import Optional, List, Dict, Union, TYPE_CHECKING

from finceptquantlib.core.foundations.types import (
    Rate, Currency
)

if TYPE_CHECKING:
    from finceptquantlib.curves.curve import Curve


# =============================================================================
# Enums
# =============================================================================

class CompoundingMethod(Enum):
    """Method for compounding rates."""
    NONE = auto()           # No compounding (simple)
    FLAT = auto()           # Flat compounding (spread not compounded)
    SPREAD_EXCLUSIVE = auto()  # Spread added after compounding
    SPREAD_INCLUSIVE = auto()  # Spread compounded with rate


class FixingType(Enum):
    """Type of rate fixing."""
    IN_ADVANCE = auto()     # Fix at period start
    IN_ARREARS = auto()     # Fix at period end
    AVERAGED = auto()       # Average of daily fixings
    COMPOUNDED = auto()     # Compounded daily fixings


class ObservationMethod(Enum):
    """RFR observation method per ISDA definitions."""
    STANDARD = auto()           # Standard observation
    PAYMENT_DELAY = auto()      # Observation ends before payment
    OBSERVATION_SHIFT = auto()  # Shift observation period
    LOOKBACK = auto()           # Lookback without observation shift
    LOCKOUT = auto()            # Freeze rate for final days



# =============================================================================
# Base Period Protocol
# =============================================================================

class BasePeriod(ABC):
    """
    Abstract base class for all period types.

    A period represents a single payment in a leg, with:
    - Accrual dates (start, end)
    - Payment date
    - Notional amount
    - Calculated payment amount
    """

    @property
    @abstractmethod
    def accrual_start(self) -> date:
        """Start of accrual period."""

    @property
    @abstractmethod
    def accrual_end(self) -> date:
        """End of accrual period."""

    @property
    @abstractmethod
    def payment_date(self) -> date:
        """Date when payment is made."""

    @property
    @abstractmethod
    def notional(self) -> float:
        """Notional amount for the period."""

    @property
    @abstractmethod
    def currency(self) -> Currency:
        """Currency of the payment."""

    @abstractmethod
    def cashflow(
        self,
        curve: Optional[Curve] = None,
        fixing_rate: Optional[float] = None
    ) -> float:
        """
        Calculate the cashflow amount for this period.

        Args:
            curve: Discount/forecast curve (for floating periods)
            fixing_rate: Override fixing rate

        Returns:
            Cashflow amount (positive = receive, negative = pay)
        """

    @abstractmethod
    def npv(
        self,
        discount_curve: Curve,
        forecast_curve: Optional[Curve] = None,
        valuation_date: Optional[date] = None
    ) -> float:
        """
        Calculate net present value of the period.

        Args:
            discount_curve: Curve for discounting
            forecast_curve: Curve for rate forecasting (floating periods)
            valuation_date: Date for discounting (default: curve base date)

        Returns:
            NPV of the period
        """

    @abstractmethod
    def accrual_fraction(self) -> float:
        """Day count fraction for the accrual period."""


# =============================================================================
# Day Count Fraction Calculator (Internal)
# =============================================================================

def _dcf_act360(start: date, end: date) -> float:
    """ACT/360 day count fraction."""
    return (end - start).days / 360.0


def _dcf_act365(start: date, end: date) -> float:
    """ACT/365 day count fraction."""
    return (end - start).days / 365.0


def _dcf_30360(start: date, end: date) -> float:
    """30/360 day count fraction."""
    d1 = min(start.day, 30)
    d2 = min(end.day, 30) if d1 == 30 else end.day

    return (
        360 * (end.year - start.year) +
        30 * (end.month - start.month) +
        (d2 - d1)
    ) / 360.0


def _dcf_actact_isda(start: date, end: date) -> float:
    """ACT/ACT ISDA day count fraction.

    Accounts for leap years by splitting the accrual period at year boundaries.
    Each portion is divided by the number of days in that year (365 or 366).
    """
    if start >= end:
        return 0.0

    total = 0.0
    current = start

    while current.year < end.year:
        year_end = date(current.year + 1, 1, 1)
        days_in_year = 366 if (
            current.year % 4 == 0 and
            (current.year % 100 != 0 or current.year % 400 == 0)
        ) else 365
        total += (year_end - current).days / days_in_year
        current = year_end

    # Final partial year
    days_in_year = 366 if (
        end.year % 4 == 0 and
        (end.year % 100 != 0 or end.year % 400 == 0)
    ) else 365
    total += (end - current).days / days_in_year

    return total


def _resolve_day_count(dc_str: str) -> str:
    """Normalize day count convention string to canonical form."""
    dc = dc_str.upper().replace(" ", "").replace("/", "")
    if dc in ("ACT360", "ACTUAL360"):
        return "ACT360"
    elif dc in ("ACT365", "ACT365F", "ACT365FIXED", "ACTUAL365"):
        return "ACT365"
    elif dc in ("30360", "THIRTY360", "360360"):
        return "30360"
    elif dc in ("ACTACT", "ACTACTISDA", "ACTUALACTUAL"):
        return "ACTACT"
    return "ACT360"  # default


def _compute_dcf(dc_key: str, start: date, end: date) -> float:
    """Compute day count fraction from resolved key."""
    if dc_key == "ACT360":
        return _dcf_act360(start, end)
    elif dc_key == "ACT365":
        return _dcf_act365(start, end)
    elif dc_key == "30360":
        return _dcf_30360(start, end)
    elif dc_key == "ACTACT":
        return _dcf_actact_isda(start, end)
    return _dcf_act360(start, end)


# =============================================================================
# Fixed Coupon Period
# =============================================================================

@dataclass
class FixedCouponPeriod(BasePeriod):
    """
    Single fixed rate coupon payment.

    Attributes:
        start: Accrual period start
        end: Accrual period end
        payment: Payment date
        rate: Fixed coupon rate
        notional_amount: Notional principal
        ccy: Currency
        day_count: Day count convention ("ACT/360", "ACT/365", "30/360")
    """
    start: date
    end: date
    payment: date
    rate: Rate
    notional_amount: float
    ccy: Currency
    day_count: str = "ACT/360"

    _dc_key: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.rate, (float, int, str)):
            self.rate = Rate(self.rate)
        if isinstance(self.ccy, str):
            self.ccy = Currency.from_string(self.ccy)
        self._dc_key = _resolve_day_count(self.day_count)

    @property
    def accrual_start(self) -> date:
        return self.start

    @property
    def accrual_end(self) -> date:
        return self.end

    @property
    def payment_date(self) -> date:
        return self.payment

    @property
    def notional(self) -> float:
        return self.notional_amount

    @property
    def currency(self) -> Currency:
        return self.ccy

    def accrual_fraction(self) -> float:
        """Calculate day count fraction."""
        return _compute_dcf(self._dc_key, self.start, self.end)

    def cashflow(
        self,
        curve: Optional[Curve] = None,
        fixing_rate: Optional[float] = None
    ) -> float:
        """Calculate fixed coupon cashflow."""
        dcf = self.accrual_fraction()
        return self.notional_amount * self.rate.decimal * dcf

    def npv(
        self,
        discount_curve: Curve,
        forecast_curve: Optional[Curve] = None,
        valuation_date: Optional[date] = None
    ) -> float:
        """Calculate NPV of fixed coupon."""
        cf = self.cashflow()
        df = discount_curve.discount_factor(self.payment)
        return cf * df

    def __repr__(self) -> str:
        return (
            f"FixedCouponPeriod({self.start} -> {self.end}, "
            f"rate={self.rate}, notional={self.notional_amount:,.0f})"
        )


# =============================================================================
# Floating Coupon Period
# =============================================================================

@dataclass
class FloatCouponPeriod(BasePeriod):
    """
    Single floating rate coupon payment.

    Attributes:
        start: Accrual period start
        end: Accrual period end
        payment: Payment date
        fixing_date: Date when rate is fixed
        spread: Spread over index rate
        notional_amount: Notional principal
        ccy: Currency
        day_count: Day count convention
        fixing_type: When rate is fixed (in advance/arrears)
        index_name: Name of the reference index (e.g., "SOFR", "EURIBOR")
    """
    start: date
    end: date
    payment: date
    fixing_date: date
    notional_amount: float
    ccy: Currency
    spread: Rate = field(default_factory=lambda: Rate(0.0))
    day_count: str = "ACT/360"
    fixing_type: FixingType = FixingType.IN_ADVANCE
    index_name: str = ""
    _fixed_rate: Optional[float] = field(default=None, repr=False)
    _dc_key: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.spread, (float, int, str)):
            self.spread = Rate(self.spread)
        if isinstance(self.ccy, str):
            self.ccy = Currency.from_string(self.ccy)
        self._dc_key = _resolve_day_count(self.day_count)

    @property
    def accrual_start(self) -> date:
        return self.start

    @property
    def accrual_end(self) -> date:
        return self.end

    @property
    def payment_date(self) -> date:
        return self.payment

    @property
    def notional(self) -> float:
        return self.notional_amount

    @property
    def currency(self) -> Currency:
        return self.ccy

    @property
    def is_fixed(self) -> bool:
        """Check if the rate has been fixed."""
        return self._fixed_rate is not None

    def set_fixing(self, rate: Union[float, Rate]) -> None:
        """Set the fixing rate."""
        if isinstance(rate, Rate):
            self._fixed_rate = rate.decimal
        else:
            self._fixed_rate = float(rate)

    def accrual_fraction(self) -> float:
        """Calculate day count fraction."""
        return _compute_dcf(self._dc_key, self.start, self.end)

    def forward_rate(self, curve: Curve) -> float:
        """Calculate forward rate from curve."""
        return curve.forward_rate(self.start, self.end)

    def cashflow(
        self,
        curve: Optional[Curve] = None,
        fixing_rate: Optional[float] = None
    ) -> float:
        """
        Calculate floating coupon cashflow.

        Priority for rate determination:
        1. fixing_rate parameter
        2. _fixed_rate if set
        3. Forward rate from curve
        """
        if fixing_rate is not None:
            rate = fixing_rate
        elif self._fixed_rate is not None:
            rate = self._fixed_rate
        elif curve is not None:
            rate = self.forward_rate(curve)
        else:
            raise ValueError(
                "Cannot calculate cashflow: no fixing rate, fixed rate, or curve provided"
            )

        dcf = self.accrual_fraction()
        total_rate = rate + self.spread.decimal
        return self.notional_amount * total_rate * dcf

    def npv(
        self,
        discount_curve: Curve,
        forecast_curve: Optional[Curve] = None,
        valuation_date: Optional[date] = None
    ) -> float:
        """Calculate NPV of floating coupon."""
        fc = forecast_curve if forecast_curve is not None else discount_curve
        cf = self.cashflow(curve=fc)
        df = discount_curve.discount_factor(self.payment)
        return cf * df

    def __repr__(self) -> str:
        rate_str = f"{self._fixed_rate:.4%}" if self._fixed_rate else "unfixed"
        return (
            f"FloatCouponPeriod({self.start} -> {self.end}, "
            f"rate={rate_str}, spread={self.spread}, notional={self.notional_amount:,.0f})"
        )


# =============================================================================
# Compounded RFR Period
# =============================================================================

@dataclass
class CompoundedRFRPeriod(BasePeriod):
    """
    Compounded Risk-Free Rate period (SOFR, SONIA, ESTR, etc.).

    Handles various compounding conventions per ISDA definitions.

    Attributes:
        start: Accrual period start
        end: Accrual period end
        payment: Payment date
        observation_start: Start of observation period
        observation_end: End of observation period
        spread: Spread over compounded rate
        notional_amount: Notional principal
        ccy: Currency
        day_count: Day count convention
        observation_method: ISDA observation method
        lookback_days: Days for lookback/lockout
        payment_delay_days: Days payment is delayed from accrual end
    """
    start: date
    end: date
    payment: date
    notional_amount: float
    ccy: Currency
    observation_start: Optional[date] = None
    observation_end: Optional[date] = None
    spread: Rate = field(default_factory=lambda: Rate(0.0))
    day_count: str = "ACT/360"
    observation_method: ObservationMethod = ObservationMethod.STANDARD
    lookback_days: int = 0
    payment_delay_days: int = 0
    compounding_method: CompoundingMethod = CompoundingMethod.SPREAD_EXCLUSIVE
    _compounded_rate: Optional[float] = field(default=None, repr=False)
    _daily_rates: Optional[Dict[date, float]] = field(default=None, repr=False)
    _dc_key: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.spread, (float, int, str)):
            self.spread = Rate(self.spread)
        if isinstance(self.ccy, str):
            self.ccy = Currency.from_string(self.ccy)
        self._dc_key = _resolve_day_count(self.day_count)

        # Set observation period based on method
        if self.observation_start is None:
            if self.observation_method == ObservationMethod.OBSERVATION_SHIFT:
                self.observation_start = self.start - timedelta(days=self.lookback_days)
            elif self.observation_method == ObservationMethod.LOOKBACK:
                self.observation_start = self.start - timedelta(days=self.lookback_days)
            else:
                self.observation_start = self.start

        if self.observation_end is None:
            if self.observation_method == ObservationMethod.OBSERVATION_SHIFT:
                self.observation_end = self.end - timedelta(days=self.lookback_days)
            elif self.observation_method == ObservationMethod.PAYMENT_DELAY:
                self.observation_end = self.end
            else:
                self.observation_end = self.end

    @property
    def accrual_start(self) -> date:
        return self.start

    @property
    def accrual_end(self) -> date:
        return self.end

    @property
    def payment_date(self) -> date:
        return self.payment

    @property
    def notional(self) -> float:
        return self.notional_amount

    @property
    def currency(self) -> Currency:
        return self.ccy

    def accrual_fraction(self) -> float:
        """Calculate day count fraction."""
        return _compute_dcf(self._dc_key, self.start, self.end)

    def set_daily_rates(self, rates: Dict[date, float]) -> None:
        """Set daily fixing rates for compounding."""
        self._daily_rates = rates
        self._compounded_rate = None  # Clear cached rate

    def compound_daily_rates(
        self,
        daily_rates: Optional[Dict[date, float]] = None
    ) -> float:
        """
        Compound daily RFR rates.

        Args:
            daily_rates: Dictionary of date -> rate (decimal)

        Returns:
            Compounded rate for the period
        """
        rates = daily_rates if daily_rates is not None else self._daily_rates
        if rates is None:
            raise ValueError("No daily rates available for compounding")

        # Determine dates to compound
        obs_start = self.observation_start or self.start
        obs_end = self.observation_end or self.end

        # Calculate number of days for denominator
        dc = self.day_count.upper().replace(" ", "").replace("/", "")
        year_basis = 360.0 if "360" in dc else 365.0

        # Compound the rates
        product = 1.0
        current = obs_start

        while current < obs_end:
            # Handle lockout: use last rate for final days
            if (self.observation_method == ObservationMethod.LOCKOUT and
                (obs_end - current).days <= self.lookback_days):
                # Use the rate from lockout_days before end
                lockout_date = obs_end - timedelta(days=self.lookback_days + 1)
                rate = rates.get(lockout_date, 0.0)
            else:
                rate = rates.get(current, 0.0)

            # Daily factor depends on compounding method
            if self.compounding_method == CompoundingMethod.SPREAD_INCLUSIVE:
                # Include spread in daily compounding
                daily_factor = 1.0 + (rate + self.spread.decimal) / year_basis
            else:
                daily_factor = 1.0 + rate / year_basis
            product *= daily_factor
            current += timedelta(days=1)

        # Convert product to annualized rate
        total_days = (obs_end - obs_start).days
        if total_days > 0:
            compounded = (product - 1.0) * year_basis / total_days
        else:
            compounded = 0.0

        self._compounded_rate = compounded
        return compounded

    def cashflow(
        self,
        curve: Optional[Curve] = None,
        fixing_rate: Optional[float] = None
    ) -> float:
        """Calculate compounded RFR cashflow."""
        if fixing_rate is not None:
            rate = fixing_rate
        elif self._compounded_rate is not None:
            rate = self._compounded_rate
        elif self._daily_rates is not None:
            rate = self.compound_daily_rates()
        elif curve is not None:
            # Use curve to estimate compounded rate
            rate = curve.forward_rate(self.start, self.end)
        else:
            raise ValueError("No rate available for calculation")

        dcf = self.accrual_fraction()

        # Apply spread based on compounding method
        if self.compounding_method == CompoundingMethod.SPREAD_EXCLUSIVE:
            # Spread added after compounding: compounded_rate + spread
            total_rate = rate + self.spread.decimal
        elif self.compounding_method == CompoundingMethod.FLAT:
            # Flat compounding: compound (r) then add spread * dcf separately
            # Effective amount = notional * (compounded_r * dcf + spread * dcf)
            # Same formula but semantically different — rate is pure compounded
            total_rate = rate + self.spread.decimal
        else:
            # SPREAD_INCLUSIVE: spread was compounded together with rate
            # The compounded rate already includes the spread in daily factors
            # so don't add spread again
            total_rate = rate

        return self.notional_amount * total_rate * dcf

    def npv(
        self,
        discount_curve: Curve,
        forecast_curve: Optional[Curve] = None,
        valuation_date: Optional[date] = None
    ) -> float:
        """Calculate NPV of compounded RFR period."""
        fc = forecast_curve if forecast_curve is not None else discount_curve
        cf = self.cashflow(curve=fc)
        df = discount_curve.discount_factor(self.payment)
        return cf * df

    def __repr__(self) -> str:
        rate_str = f"{self._compounded_rate:.4%}" if self._compounded_rate else "uncompounded"
        return (
            f"CompoundedRFRPeriod({self.start} -> {self.end}, "
            f"rate={rate_str}, method={self.observation_method.name})"
        )


# =============================================================================
# Inflation Period
# =============================================================================

@dataclass
class InflationPeriod(BasePeriod):
    """
    Inflation-linked payment period.

    Attributes:
        start: Accrual period start
        end: Accrual period end
        payment: Payment date
        base_index: Base inflation index value
        reference_date: Reference date for index lookup
        notional_amount: Notional principal
        ccy: Currency
        index_lag_months: Months lag for index (typically 2 or 3)
        interpolation: Whether to interpolate index values
    """
    start: date
    end: date
    payment: date
    notional_amount: float
    ccy: Currency
    base_index: float
    reference_date: Optional[date] = None
    index_lag_months: int = 3
    interpolation: bool = True
    day_count: str = "ACT/ACT"
    _end_index: Optional[float] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.ccy, str):
            self.ccy = Currency.from_string(self.ccy)

    @property
    def accrual_start(self) -> date:
        return self.start

    @property
    def accrual_end(self) -> date:
        return self.end

    @property
    def payment_date(self) -> date:
        return self.payment

    @property
    def notional(self) -> float:
        return self.notional_amount

    @property
    def currency(self) -> Currency:
        return self.ccy

    def accrual_fraction(self) -> float:
        """Calculate day count fraction."""
        return _dcf_act365(self.start, self.end)

    def index_reference_date(self, dt: date) -> date:
        """Calculate reference date for index lookup with lag."""
        month = dt.month - self.index_lag_months
        year = dt.year
        while month < 1:
            month += 12
            year -= 1
        return date(year, month, 1)

    def set_end_index(self, index_value: float) -> None:
        """Set the end-period index value."""
        self._end_index = index_value

    def index_ratio(self, end_index: Optional[float] = None) -> float:
        """Calculate index ratio (end / base)."""
        idx = end_index if end_index is not None else self._end_index
        if idx is None:
            raise ValueError("End index value not set")
        return idx / self.base_index

    def cashflow(
        self,
        curve: Optional[Curve] = None,
        fixing_rate: Optional[float] = None
    ) -> float:
        """
        Calculate inflation-linked cashflow.

        For inflation periods, the cashflow is:
        notional * (end_index / base_index - 1)
        """
        if self._end_index is None:
            raise ValueError("End index value not set")

        inflation_adjustment = self.index_ratio() - 1.0
        return self.notional_amount * inflation_adjustment

    def npv(
        self,
        discount_curve: Curve,
        forecast_curve: Optional[Curve] = None,
        valuation_date: Optional[date] = None
    ) -> float:
        """Calculate NPV of inflation period."""
        cf = self.cashflow()
        df = discount_curve.discount_factor(self.payment)
        return cf * df

    def __repr__(self) -> str:
        return (
            f"InflationPeriod({self.start} -> {self.end}, "
            f"base_index={self.base_index:.2f}, notional={self.notional_amount:,.0f})"
        )


# =============================================================================
# Principal Exchange
# =============================================================================

class ExchangeType(Enum):
    """Type of principal exchange."""
    INITIAL = auto()      # At start of swap
    FINAL = auto()        # At maturity
    INTERMEDIATE = auto() # During swap life (amortization)


@dataclass
class PrincipalExchange(BasePeriod):
    """
    Principal exchange payment.

    Used for:
    - Initial/final exchanges in cross-currency swaps
    - Amortization payments
    - Bullet repayments
    """
    exchange_date: date
    amount: float
    ccy: Currency
    exchange_type: ExchangeType = ExchangeType.FINAL
    pay_receive: int = 1  # 1 = receive, -1 = pay

    def __post_init__(self) -> None:
        if isinstance(self.ccy, str):
            self.ccy = Currency.from_string(self.ccy)

    @property
    def accrual_start(self) -> date:
        return self.exchange_date

    @property
    def accrual_end(self) -> date:
        return self.exchange_date

    @property
    def payment_date(self) -> date:
        return self.exchange_date

    @property
    def notional(self) -> float:
        return abs(self.amount)

    @property
    def currency(self) -> Currency:
        return self.ccy

    def accrual_fraction(self) -> float:
        return 0.0

    def cashflow(
        self,
        curve: Optional[Curve] = None,
        fixing_rate: Optional[float] = None
    ) -> float:
        """Return principal exchange amount."""
        return self.amount * self.pay_receive

    def npv(
        self,
        discount_curve: Curve,
        forecast_curve: Optional[Curve] = None,
        valuation_date: Optional[date] = None
    ) -> float:
        """Calculate NPV of principal exchange."""
        cf = self.cashflow()
        df = discount_curve.discount_factor(self.exchange_date)
        return cf * df

    def __repr__(self) -> str:
        direction = "receive" if self.pay_receive > 0 else "pay"
        return (
            f"PrincipalExchange({self.exchange_date}, {direction} "
            f"{self.ccy.value} {abs(self.amount):,.0f})"
        )


# =============================================================================
# Fee/Premium Payment
# =============================================================================

@dataclass
class FeePayment(BasePeriod):
    """
    One-time fee or premium payment.

    Used for:
    - Upfront premiums
    - Arrangement fees
    - Settlement amounts
    """
    fee_date: date
    amount: float
    ccy: Currency
    description: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.ccy, str):
            self.ccy = Currency.from_string(self.ccy)

    @property
    def accrual_start(self) -> date:
        return self.fee_date

    @property
    def accrual_end(self) -> date:
        return self.fee_date

    @property
    def payment_date(self) -> date:
        return self.fee_date

    @property
    def notional(self) -> float:
        return abs(self.amount)

    @property
    def currency(self) -> Currency:
        return self.ccy

    def accrual_fraction(self) -> float:
        return 0.0

    def cashflow(
        self,
        curve: Optional[Curve] = None,
        fixing_rate: Optional[float] = None
    ) -> float:
        return self.amount

    def npv(
        self,
        discount_curve: Curve,
        forecast_curve: Optional[Curve] = None,
        valuation_date: Optional[date] = None
    ) -> float:
        cf = self.cashflow()
        df = discount_curve.discount_factor(self.fee_date)
        return cf * df

    def __repr__(self) -> str:
        return f"FeePayment({self.fee_date}, {self.ccy.value} {self.amount:,.0f})"


# =============================================================================
# Credit Protection Period (for CDS)
# =============================================================================

@dataclass
class CreditProtectionPeriod(BasePeriod):
    """
    Credit protection period for CDS.

    Represents the contingent payment in case of default.
    """
    start: date
    end: date
    notional_amount: float
    ccy: Currency
    recovery_rate: float = 0.4
    _default_probability: Optional[float] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.ccy, str):
            self.ccy = Currency.from_string(self.ccy)

    @property
    def accrual_start(self) -> date:
        return self.start

    @property
    def accrual_end(self) -> date:
        return self.end

    @property
    def payment_date(self) -> date:
        return self.end  # Protection payment at end of period

    @property
    def notional(self) -> float:
        return self.notional_amount

    @property
    def currency(self) -> Currency:
        return self.ccy

    @property
    def loss_given_default(self) -> float:
        return 1.0 - self.recovery_rate

    def accrual_fraction(self) -> float:
        return _dcf_act360(self.start, self.end)

    def set_default_probability(self, prob: float) -> None:
        """Set default probability for the period."""
        self._default_probability = prob

    def expected_loss(self, default_prob: Optional[float] = None) -> float:
        """Calculate expected loss for the period."""
        prob = default_prob if default_prob is not None else self._default_probability
        if prob is None:
            raise ValueError("Default probability not set")
        return self.notional_amount * self.loss_given_default * prob

    def cashflow(
        self,
        curve: Optional[Curve] = None,
        fixing_rate: Optional[float] = None
    ) -> float:
        """Expected protection payment."""
        if self._default_probability is None:
            return 0.0
        return self.expected_loss()

    def npv(
        self,
        discount_curve: Curve,
        forecast_curve: Optional[Curve] = None,
        valuation_date: Optional[date] = None
    ) -> float:
        cf = self.cashflow()
        df = discount_curve.discount_factor(self.end)
        return cf * df

    def __repr__(self) -> str:
        return (
            f"CreditProtectionPeriod({self.start} -> {self.end}, "
            f"notional={self.notional_amount:,.0f}, RR={self.recovery_rate:.0%})"
        )


# =============================================================================
# Period List Type
# =============================================================================

# Type alias for a list of periods
PeriodList = List[BasePeriod]


def total_cashflows(periods: PeriodList, curve: Optional[Curve] = None) -> float:
    """Calculate total cashflows from a list of periods."""
    return sum(p.cashflow(curve) for p in periods)


def total_npv(
    periods: PeriodList,
    discount_curve: Curve,
    forecast_curve: Optional[Curve] = None
) -> float:
    """Calculate total NPV from a list of periods."""
    return sum(p.npv(discount_curve, forecast_curve) for p in periods)
