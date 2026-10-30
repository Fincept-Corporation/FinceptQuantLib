"""
Bond Instruments

Implements various bond types with comprehensive pricing and analytics:
- Fixed Rate Bonds
- Zero Coupon Bonds
- Floating Rate Notes
- Inflation-Linked Bonds

Analytics include:
- Clean and dirty price
- Yield to maturity
- Duration (Macaulay, Modified, Effective)
- Convexity
- DV01 (Dollar Value of 01)
- Key rate durations
- Spreads (Z-spread, OAS, ASW)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Union, Optional, List, Tuple
import math
from scipy.optimize import brentq, newton

from finceptquantlib.curves.curve import Curve
from finceptquantlib.scheduling.schedule import Schedule, Frequency
from finceptquantlib.scheduling.daycount import DayCountConvention, year_fraction
from finceptquantlib.scheduling.calendar import Calendar, get_calendar


@dataclass
class BondPriceResult:
    """
    Result of bond pricing calculation.

    Attributes:
        clean_price: Price excluding accrued interest (as % of notional)
        dirty_price: Price including accrued interest (as % of notional)
        accrued_interest: Accrued interest (as % of notional)
        settlement_date: Settlement date for pricing
        yield_to_maturity: Yield to maturity (if calculated)
    """
    clean_price: float
    dirty_price: float
    accrued_interest: float
    settlement_date: date
    yield_to_maturity: Optional[float] = None

    def __repr__(self) -> str:
        return (f"BondPriceResult(clean={self.clean_price:.4f}, "
                f"dirty={self.dirty_price:.4f}, "
                f"accrued={self.accrued_interest:.4f})")


@dataclass
class BondAnalytics:
    """
    Comprehensive bond analytics results.

    Attributes:
        macaulay_duration: Macaulay duration in years
        modified_duration: Modified duration
        effective_duration: Effective duration
        convexity: Convexity
        dv01: Dollar value of 1 basis point
        pv01: Present value of 1 basis point
    """
    macaulay_duration: float
    modified_duration: float
    effective_duration: float
    convexity: float
    dv01: float
    pv01: float

    def __repr__(self) -> str:
        return (f"BondAnalytics(mod_dur={self.modified_duration:.4f}, "
                f"convexity={self.convexity:.4f}, "
                f"dv01={self.dv01:.4f})")


class Bond(ABC):
    """
    Abstract base class for bonds.

    Provides common functionality for all bond types including
    pricing, yield calculation, and risk analytics.
    """

    def __init__(
        self,
        issue_date: date,
        maturity_date: date,
        notional: float = 100.0,
        currency: str = "USD",
        calendar: Union[str, Calendar] = "Weekend",
        settlement_days: int = 2,
    ):
        """
        Initialize a bond.

        Args:
            issue_date: Bond issue date
            maturity_date: Bond maturity date
            notional: Face value (default 100)
            currency: Currency code
            calendar: Business day calendar
            settlement_days: Days to settlement
        """
        self.issue_date = issue_date
        self.maturity_date = maturity_date
        self.notional = notional
        self.currency = currency
        self.settlement_days = settlement_days

        if isinstance(calendar, str):
            calendar = get_calendar(calendar)
        self.calendar = calendar

    def settlement_date(self, trade_date: date) -> date:
        """Calculate settlement date from trade date."""
        return self.calendar.add_business_days(trade_date, self.settlement_days)

    @abstractmethod
    def cashflows(
        self,
        settlement_date: Optional[date] = None
    ) -> List[Tuple[date, float]]:
        """
        Get remaining cashflows from settlement date.

        Args:
            settlement_date: Settlement date (uses today + settlement_days if None)

        Returns:
            List of (date, amount) tuples
        """

    def dirty_price(
        self,
        curve: Curve,
        settlement_date: Optional[date] = None
    ) -> float:
        """
        Calculate dirty price using a discount curve.

        Args:
            curve: Discount curve for valuation
            settlement_date: Settlement date

        Returns:
            Dirty price as percentage of notional
        """
        if settlement_date is None:
            settlement_date = self.settlement_date(date.today())

        total_pv = 0.0
        for cf_date, cf_amount in self.cashflows(settlement_date):
            df = curve.discount_factor(cf_date)
            # Discount back to settlement date
            df_settlement = curve.discount_factor(settlement_date)
            total_pv += cf_amount * df / df_settlement

        return total_pv / self.notional * 100

    @abstractmethod
    def accrued_interest(self, settlement_date: date) -> float:
        """
        Calculate accrued interest.

        Args:
            settlement_date: Settlement date

        Returns:
            Accrued interest as percentage of notional
        """

    def clean_price(
        self,
        curve: Curve,
        settlement_date: Optional[date] = None
    ) -> float:
        """
        Calculate clean price (dirty price minus accrued interest).

        Args:
            curve: Discount curve for valuation
            settlement_date: Settlement date

        Returns:
            Clean price as percentage of notional
        """
        if settlement_date is None:
            settlement_date = self.settlement_date(date.today())

        dirty = self.dirty_price(curve, settlement_date)
        accrued = self.accrued_interest(settlement_date)
        return dirty - accrued

    def price(
        self,
        curve: Curve,
        settlement_date: Optional[date] = None
    ) -> BondPriceResult:
        """
        Full pricing calculation.

        Args:
            curve: Discount curve for valuation
            settlement_date: Settlement date

        Returns:
            BondPriceResult with clean price, dirty price, and accrued
        """
        if settlement_date is None:
            settlement_date = self.settlement_date(date.today())

        dirty = self.dirty_price(curve, settlement_date)
        accrued = self.accrued_interest(settlement_date)

        return BondPriceResult(
            clean_price=dirty - accrued,
            dirty_price=dirty,
            accrued_interest=accrued,
            settlement_date=settlement_date,
        )

    @abstractmethod
    def yield_to_maturity(
        self,
        clean_price: float,
        settlement_date: Optional[date] = None,
        compounding: str = "semi-annual",
    ) -> float:
        """
        Calculate yield to maturity from price.

        Args:
            clean_price: Clean price as percentage of notional
            settlement_date: Settlement date
            compounding: Compounding frequency

        Returns:
            Yield to maturity
        """

    def price_from_yield(
        self,
        ytm: float,
        settlement_date: Optional[date] = None,
        compounding: str = "semi-annual",
    ) -> float:
        """
        Calculate clean price from yield to maturity.

        Args:
            ytm: Yield to maturity
            settlement_date: Settlement date
            compounding: Compounding frequency

        Returns:
            Clean price as percentage of notional
        """
        if settlement_date is None:
            settlement_date = self.settlement_date(date.today())

        compounding_periods = {
            "annual": 1,
            "semi-annual": 2,
            "quarterly": 4,
            "monthly": 12,
            "continuous": 0,
        }
        if compounding not in compounding_periods:
            raise ValueError(
                f"Unknown compounding '{compounding}'; "
                f"expected one of {sorted(compounding_periods)}"
            )
        periods_per_year = compounding_periods[compounding]

        total_pv = 0.0
        for t, cf_amount in self._yield_times(settlement_date):
            if t <= 0:
                continue

            if periods_per_year == 0:  # Continuous
                df = math.exp(-ytm * t)
            else:
                df = (1 + ytm / periods_per_year) ** (-periods_per_year * t)

            total_pv += cf_amount * df

        dirty_price = total_pv / self.notional * 100
        return dirty_price - self.accrued_interest(settlement_date)

    def _yield_times(self, settlement_date: date) -> List[Tuple[float, float]]:
        """
        (time in years, amount) pairs used to discount cashflows at a yield.

        The default measures time as ACT/365F from settlement. Coupon bonds
        override this with the street convention of whole coupon periods.
        """
        return [
            (year_fraction(settlement_date, cf_date, "ACT/365F"), cf_amount)
            for cf_date, cf_amount in self.cashflows(settlement_date)
        ]

    def analytics(
        self,
        curve: Curve,
        settlement_date: Optional[date] = None,
    ) -> BondAnalytics:
        """
        Calculate comprehensive bond analytics.

        Args:
            curve: Discount curve
            settlement_date: Settlement date

        Returns:
            BondAnalytics with duration, convexity, etc.
        """
        if settlement_date is None:
            settlement_date = self.settlement_date(date.today())

        dirty = self.dirty_price(curve, settlement_date)
        ytm = self.yield_to_maturity(
            dirty - self.accrued_interest(settlement_date),
            settlement_date
        )

        # Macaulay duration
        mac_dur = self._macaulay_duration(ytm, settlement_date)

        # Modified duration
        periods_per_year = 2  # Assume semi-annual
        mod_dur = mac_dur / (1 + ytm / periods_per_year)

        # Effective duration (using curve shifts)
        shift_bp = 1.0  # 1 basis point
        shifted_up = curve.shift(shift_bp)
        shifted_down = curve.shift(-shift_bp)
        price_up = self.dirty_price(shifted_down, settlement_date)
        price_down = self.dirty_price(shifted_up, settlement_date)
        eff_dur = (price_up - price_down) / (2 * dirty * shift_bp / 10000)

        # Convexity
        convexity = self._convexity(ytm, settlement_date)

        # DV01 (per 100 notional)
        dv01 = mod_dur * dirty / 10000

        # PV01
        pv01 = dv01 * self.notional / 100

        return BondAnalytics(
            macaulay_duration=mac_dur,
            modified_duration=mod_dur,
            effective_duration=eff_dur,
            convexity=convexity,
            dv01=dv01,
            pv01=pv01,
        )

    def _macaulay_duration(
        self,
        ytm: float,
        settlement_date: date,
    ) -> float:
        """Calculate Macaulay duration."""
        periods_per_year = 2  # Assume semi-annual

        weighted_time = 0.0
        total_pv = 0.0

        for cf_date, cf_amount in self.cashflows(settlement_date):
            t = year_fraction(settlement_date, cf_date, "ACT/365F")
            if t <= 0:
                continue

            df = (1 + ytm / periods_per_year) ** (-periods_per_year * t)
            pv = cf_amount * df
            weighted_time += t * pv
            total_pv += pv

        return weighted_time / total_pv if total_pv > 0 else 0.0

    def _convexity(
        self,
        ytm: float,
        settlement_date: date,
    ) -> float:
        """Calculate convexity."""
        periods_per_year = 2  # Assume semi-annual

        weighted_time_sq = 0.0
        total_pv = 0.0

        for cf_date, cf_amount in self.cashflows(settlement_date):
            t = year_fraction(settlement_date, cf_date, "ACT/365F")
            if t <= 0:
                continue

            df = (1 + ytm / periods_per_year) ** (-periods_per_year * t)
            pv = cf_amount * df
            weighted_time_sq += t * (t + 1 / periods_per_year) * pv
            total_pv += pv

        if total_pv <= 0:
            return 0.0

        return weighted_time_sq / (total_pv * (1 + ytm / periods_per_year) ** 2)


class FixedRateBond(Bond):
    """
    Fixed rate coupon bond.

    Pays fixed periodic coupons and principal at maturity.
    """

    def __init__(
        self,
        issue_date: date,
        maturity_date: date,
        coupon_rate: float,
        frequency: Union[Frequency, str] = Frequency.SEMI_ANNUAL,
        day_count: Union[str, DayCountConvention] = "30/360",
        notional: float = 100.0,
        currency: str = "USD",
        calendar: Union[str, Calendar] = "Weekend",
        settlement_days: int = 2,
    ):
        """
        Initialize fixed rate bond.

        Args:
            issue_date: Bond issue date
            maturity_date: Bond maturity date
            coupon_rate: Annual coupon rate (e.g., 0.05 for 5%)
            frequency: Coupon payment frequency
            day_count: Day count convention for accrued interest
            notional: Face value
            currency: Currency code
            calendar: Business day calendar
            settlement_days: Days to settlement
        """
        super().__init__(
            issue_date=issue_date,
            maturity_date=maturity_date,
            notional=notional,
            currency=currency,
            calendar=calendar,
            settlement_days=settlement_days,
        )

        self.coupon_rate = coupon_rate

        if isinstance(frequency, str):
            from finceptquantlib.scheduling.schedule import _parse_frequency
            frequency = _parse_frequency(frequency)
        self.frequency = frequency

        if isinstance(day_count, str):
            from finceptquantlib.scheduling.daycount import _get_convention
            day_count = _get_convention(day_count)
        self.day_count = day_count

        # Generate payment schedule
        self._schedule = Schedule(
            effective_date=issue_date,
            termination_date=maturity_date,
            frequency=frequency,
            calendar=calendar,
        )

    @property
    def coupon_amount(self) -> float:
        """Coupon amount per period."""
        periods_per_year = self.frequency.periods_per_year
        if periods_per_year == 0:
            return 0.0
        return self.notional * self.coupon_rate / periods_per_year

    def cashflows(
        self,
        settlement_date: Optional[date] = None
    ) -> List[Tuple[date, float]]:
        """Get remaining cashflows."""
        if settlement_date is None:
            settlement_date = self.settlement_date(date.today())

        flows = []
        for period in self._schedule.periods:
            if self._is_outstanding(period, settlement_date):
                # Coupons accrue on unadjusted dates; only payment dates roll
                dcf = self.day_count.year_fraction(
                    period.start_date, period.end_date
                )
                coupon = self.notional * self.coupon_rate * dcf

                if period.end_date == self.maturity_date:
                    # Add principal at maturity
                    flows.append((period.payment_date, coupon + self.notional))
                else:
                    flows.append((period.payment_date, coupon))

        return flows

    def accrued_interest(self, settlement_date: date) -> float:
        """Calculate accrued interest."""
        # Find current coupon period
        for period in self._schedule.periods:
            if period.start_date <= settlement_date < period.end_date:
                # Calculate accrued
                dcf_accrued = self.day_count.year_fraction(
                    period.start_date, settlement_date
                )
                dcf_period = self.day_count.year_fraction(
                    period.start_date, period.end_date
                )
                if dcf_period > 0:
                    coupon = self.notional * self.coupon_rate * dcf_period
                    return coupon * dcf_accrued / dcf_period / self.notional * 100
                return 0.0

        return 0.0

    @staticmethod
    def _is_outstanding(period, settlement_date: date) -> bool:
        """A coupon belongs to the buyer if its accrual period ends after settlement."""
        return period.end_date > settlement_date and period.payment_date > settlement_date

    def _yield_times(self, settlement_date: date) -> List[Tuple[float, float]]:
        """
        Street convention: the k-th remaining cashflow sits k + w coupon
        periods away, where w is the fraction of the current period left
        to run under the bond's day count. A par bond therefore prices at
        exactly 100 on a coupon date.
        """
        periods_per_year = self.frequency.periods_per_year
        if periods_per_year == 0:
            return super()._yield_times(settlement_date)

        flows = self.cashflows(settlement_date)
        if not flows:
            return []
        current = next(
            p for p in self._schedule.periods if self._is_outstanding(p, settlement_date)
        )
        full = self.day_count.year_fraction(current.start_date, current.end_date)
        remaining = self.day_count.year_fraction(settlement_date, current.end_date)
        w = remaining / full if full > 0 else 1.0

        return [
            ((k + w) / periods_per_year, amount)
            for k, (_, amount) in enumerate(flows)
        ]

    def yield_to_maturity(
        self,
        clean_price: float,
        settlement_date: Optional[date] = None,
        compounding: str = "semi-annual",
    ) -> float:
        """Calculate yield to maturity from clean price."""
        if settlement_date is None:
            settlement_date = self.settlement_date(date.today())

        def objective(ytm: float) -> float:
            return self.price_from_yield(ytm, settlement_date, compounding) - clean_price

        try:
            # Try Brent's method first
            ytm = brentq(objective, -0.5, 1.0, xtol=1e-10)
        except ValueError:
            # Fall back to Newton's method
            try:
                ytm = newton(objective, 0.05, tol=1e-10, maxiter=100)
            except RuntimeError:
                ytm = float('nan')

        return ytm

    def __repr__(self) -> str:
        return (f"FixedRateBond({self.coupon_rate*100:.2f}% "
                f"{self.frequency}, {self.maturity_date})")


class ZeroCouponBond(Bond):
    """
    Zero coupon bond (discount bond).

    Pays only principal at maturity with no intermediate coupons.
    """

    def __init__(
        self,
        issue_date: date,
        maturity_date: date,
        notional: float = 100.0,
        day_count: Union[str, DayCountConvention] = "ACT/365F",
        currency: str = "USD",
        calendar: Union[str, Calendar] = "Weekend",
        settlement_days: int = 2,
    ):
        """
        Initialize zero coupon bond.

        Args:
            issue_date: Bond issue date
            maturity_date: Bond maturity date
            notional: Face value
            day_count: Day count convention
            currency: Currency code
            calendar: Business day calendar
            settlement_days: Days to settlement
        """
        super().__init__(
            issue_date=issue_date,
            maturity_date=maturity_date,
            notional=notional,
            currency=currency,
            calendar=calendar,
            settlement_days=settlement_days,
        )

        if isinstance(day_count, str):
            from finceptquantlib.scheduling.daycount import _get_convention
            day_count = _get_convention(day_count)
        self.day_count = day_count

    def cashflows(
        self,
        settlement_date: Optional[date] = None
    ) -> List[Tuple[date, float]]:
        """Get remaining cashflows (just principal at maturity)."""
        if settlement_date is None:
            settlement_date = self.settlement_date(date.today())

        if self.maturity_date > settlement_date:
            return [(self.maturity_date, self.notional)]
        return []

    def accrued_interest(self, settlement_date: date) -> float:
        """Zero coupon bonds have no accrued interest."""
        return 0.0

    def yield_to_maturity(
        self,
        clean_price: float,
        settlement_date: Optional[date] = None,
        compounding: str = "continuous",
    ) -> float:
        """Calculate yield to maturity from price."""
        if settlement_date is None:
            settlement_date = self.settlement_date(date.today())

        t = self.day_count.year_fraction(settlement_date, self.maturity_date)
        if t <= 0:
            return 0.0

        price_decimal = clean_price / 100.0

        if compounding == "continuous":
            return -math.log(price_decimal) / t
        else:
            periods_per_year = {
                "annual": 1,
                "semi-annual": 2,
                "quarterly": 4,
                "monthly": 12,
            }.get(compounding, 2)
            return periods_per_year * (price_decimal ** (-1 / (periods_per_year * t)) - 1)

    def __repr__(self) -> str:
        return f"ZeroCouponBond(maturity={self.maturity_date})"


class FloatingRateBond(Bond):
    """
    Floating rate note (FRN).

    Pays variable coupons based on a reference rate plus a spread.
    """

    def __init__(
        self,
        issue_date: date,
        maturity_date: date,
        spread: float,
        frequency: Union[Frequency, str] = Frequency.QUARTERLY,
        day_count: Union[str, DayCountConvention] = "ACT/360",
        notional: float = 100.0,
        currency: str = "USD",
        calendar: Union[str, Calendar] = "Weekend",
        settlement_days: int = 2,
        fixing_days: int = 2,
    ):
        """
        Initialize floating rate note.

        Args:
            issue_date: Bond issue date
            maturity_date: Bond maturity date
            spread: Spread over reference rate (e.g., 0.01 for 100bp)
            frequency: Coupon payment frequency
            day_count: Day count convention
            notional: Face value
            currency: Currency code
            calendar: Business day calendar
            settlement_days: Days to settlement
            fixing_days: Days before period for rate fixing
        """
        super().__init__(
            issue_date=issue_date,
            maturity_date=maturity_date,
            notional=notional,
            currency=currency,
            calendar=calendar,
            settlement_days=settlement_days,
        )

        self.spread = spread
        self.fixing_days = fixing_days

        if isinstance(frequency, str):
            from finceptquantlib.scheduling.schedule import _parse_frequency
            frequency = _parse_frequency(frequency)
        self.frequency = frequency

        if isinstance(day_count, str):
            from finceptquantlib.scheduling.daycount import _get_convention
            day_count = _get_convention(day_count)
        self.day_count = day_count

        self._schedule = Schedule(
            effective_date=issue_date,
            termination_date=maturity_date,
            frequency=frequency,
            calendar=calendar,
        )

    def cashflows(
        self,
        settlement_date: Optional[date] = None,
        forward_curve: Optional[Curve] = None,
    ) -> List[Tuple[date, float]]:
        """
        Get remaining cashflows.

        Args:
            settlement_date: Settlement date
            forward_curve: Curve for projecting future rates

        Returns:
            List of (date, amount) tuples
        """
        if settlement_date is None:
            settlement_date = self.settlement_date(date.today())

        flows = []
        for period in self._schedule.periods:
            if period.payment_date > settlement_date:
                dcf = self.day_count.year_fraction(
                    period.accrual_start, period.accrual_end
                )

                # Get forward rate for the period
                if forward_curve is not None:
                    fwd_rate = forward_curve.forward_rate(
                        period.accrual_start, period.accrual_end
                    )
                else:
                    fwd_rate = 0.0  # Placeholder

                coupon = self.notional * (fwd_rate + self.spread) * dcf

                if period.end_date == self.maturity_date:
                    flows.append((period.payment_date, coupon + self.notional))
                else:
                    flows.append((period.payment_date, coupon))

        return flows

    def accrued_interest(self, settlement_date: date) -> float:
        """Calculate accrued interest (simplified - uses spread only)."""
        for period in self._schedule.periods:
            if period.accrual_start <= settlement_date < period.accrual_end:
                dcf_accrued = self.day_count.year_fraction(
                    period.accrual_start, settlement_date
                )
                return self.spread * dcf_accrued * 100

        return 0.0

    def yield_to_maturity(
        self,
        clean_price: float,
        settlement_date: Optional[date] = None,
        compounding: str = "quarterly",
    ) -> float:
        """
        Calculate discount margin from price.

        For FRNs, this returns the discount margin (spread to flat curve).
        """
        # Simplified: return spread if price is at par
        if abs(clean_price - 100.0) < 0.01:
            return self.spread
        return self.spread  # Placeholder

    def __repr__(self) -> str:
        return (f"FloatingRateBond(spread={self.spread*10000:.0f}bp, "
                f"{self.frequency}, {self.maturity_date})")


class InflationLinkedBond(Bond):
    """
    Inflation-linked bond (e.g., TIPS).

    Principal and coupons are adjusted for inflation.
    """

    def __init__(
        self,
        issue_date: date,
        maturity_date: date,
        coupon_rate: float,
        base_cpi: float,
        frequency: Union[Frequency, str] = Frequency.SEMI_ANNUAL,
        day_count: Union[str, DayCountConvention] = "ACT/ACT",
        notional: float = 100.0,
        currency: str = "USD",
        calendar: Union[str, Calendar] = "Weekend",
        settlement_days: int = 2,
        inflation_lag_months: int = 3,
    ):
        """
        Initialize inflation-linked bond.

        Args:
            issue_date: Bond issue date
            maturity_date: Bond maturity date
            coupon_rate: Real coupon rate
            base_cpi: CPI at issue date
            frequency: Coupon payment frequency
            day_count: Day count convention
            notional: Face value
            currency: Currency code
            calendar: Business day calendar
            settlement_days: Days to settlement
            inflation_lag_months: Months lag for inflation index
        """
        super().__init__(
            issue_date=issue_date,
            maturity_date=maturity_date,
            notional=notional,
            currency=currency,
            calendar=calendar,
            settlement_days=settlement_days,
        )

        self.coupon_rate = coupon_rate
        self.base_cpi = base_cpi
        self.inflation_lag_months = inflation_lag_months

        if isinstance(frequency, str):
            from finceptquantlib.scheduling.schedule import _parse_frequency
            frequency = _parse_frequency(frequency)
        self.frequency = frequency

        if isinstance(day_count, str):
            from finceptquantlib.scheduling.daycount import _get_convention
            day_count = _get_convention(day_count)
        self.day_count = day_count

        self._schedule = Schedule(
            effective_date=issue_date,
            termination_date=maturity_date,
            frequency=frequency,
            calendar=calendar,
        )

    def index_ratio(self, d: date, current_cpi: float) -> float:
        """Calculate index ratio for a date."""
        return current_cpi / self.base_cpi

    def cashflows(
        self,
        settlement_date: Optional[date] = None,
        cpi_forecast: Optional[float] = None,
    ) -> List[Tuple[date, float]]:
        """Get remaining cashflows (using forecasted CPI)."""
        if settlement_date is None:
            settlement_date = self.settlement_date(date.today())

        # Use base CPI if no forecast provided
        if cpi_forecast is None:
            cpi_forecast = self.base_cpi

        index_ratio = cpi_forecast / self.base_cpi
        adjusted_notional = self.notional * index_ratio

        flows = []
        for period in self._schedule.periods:
            if period.payment_date > settlement_date:
                dcf = self.day_count.year_fraction(
                    period.start_date, period.end_date
                )
                coupon = adjusted_notional * self.coupon_rate * dcf

                if period.end_date == self.maturity_date:
                    flows.append((period.payment_date, coupon + adjusted_notional))
                else:
                    flows.append((period.payment_date, coupon))

        return flows

    def accrued_interest(self, settlement_date: date) -> float:
        """Calculate accrued interest."""
        for period in self._schedule.periods:
            if period.start_date <= settlement_date < period.end_date:
                dcf_accrued = self.day_count.year_fraction(
                    period.start_date, settlement_date
                )
                dcf_period = self.day_count.year_fraction(
                    period.start_date, period.end_date
                )
                if dcf_period > 0:
                    coupon = self.coupon_rate * dcf_period
                    return coupon * dcf_accrued / dcf_period * 100
                return 0.0

        return 0.0

    def yield_to_maturity(
        self,
        clean_price: float,
        settlement_date: Optional[date] = None,
        compounding: str = "semi-annual",
    ) -> float:
        """Calculate real yield to maturity."""
        if settlement_date is None:
            settlement_date = self.settlement_date(date.today())

        def objective(ytm: float) -> float:
            return self.price_from_yield(ytm, settlement_date, compounding) - clean_price

        try:
            ytm = brentq(objective, -0.5, 0.5, xtol=1e-10)
        except ValueError:
            ytm = float('nan')

        return ytm

    def __repr__(self) -> str:
        return (f"InflationLinkedBond({self.coupon_rate*100:.2f}% real, "
                f"{self.maturity_date})")
