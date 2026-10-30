"""
Inflation Swaps

Instruments that exchange inflation-linked payments:
- Zero Coupon Inflation Swap (ZCIS)
- Year-on-Year Inflation Swap (YoY)
- Inflation-Indexed Interest Rate Swap (IIRS)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Dict, List, TYPE_CHECKING

from finceptquantlib.core.foundations.types import (
    Currency, Rate, as_tenor
)
from finceptquantlib.core.instruments.legs import CoreFixedLeg, InflationLeg

if TYPE_CHECKING:
    from finceptquantlib.curves.curve import Curve
    from finceptquantlib.curves.inflation import InflationCurve


# =============================================================================
# Zero Coupon Inflation Swap (ZCIS)
# =============================================================================

@dataclass
class ZeroCouponInflationSwap:
    """
    Zero Coupon Inflation Swap (ZCIS).

    Exchanges:
    - Fixed leg: Single compounded fixed payment at maturity
    - Inflation leg: (Final_Index / Base_Index - 1) * Notional at maturity

    The most liquid inflation derivative. Used to:
    - Trade breakeven inflation expectations
    - Hedge inflation-linked bonds
    - Construct inflation curves

    Args:
        effective_date: Start date
        termination_date: Maturity date
        notional: Notional amount
        fixed_rate: Fixed breakeven inflation rate
        currency: Currency
        inflation_index: Inflation index name (e.g., "CPI-U", "HICP")
        base_index_value: Base index value at effective date
        index_lag_months: Months lag for index lookup (typically 2-3)
        pay_inflation: True = pay inflation, receive fixed
        day_count: Day count for fixed leg
    """
    effective_date: date
    termination_date: date
    notional: float
    fixed_rate: Rate
    currency: Currency
    inflation_index: str = "CPI"
    base_index_value: Optional[float] = None
    index_lag_months: int = 3
    pay_inflation: bool = True
    day_count: str = "ACT/ACT"
    interpolation: bool = True

    _final_index_value: Optional[float] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.fixed_rate, (float, int, str)):
            self.fixed_rate = Rate(self.fixed_rate)
        if isinstance(self.currency, str):
            self.currency = Currency.from_string(self.currency)

    @property
    def tenor_years(self) -> float:
        """Swap tenor in years."""
        return (self.termination_date - self.effective_date).days / 365.0

    @property
    def tenor(self) -> str:
        """Swap tenor as string."""
        years = self.tenor_years
        if years >= 1:
            return f"{int(round(years))}Y"
        return f"{int(round(years * 12))}M"

    def set_final_index(self, value: float) -> None:
        """Set the final index value."""
        self._final_index_value = value

    def index_ratio(self, final_index: Optional[float] = None) -> float:
        """Calculate index ratio (final / base)."""
        final = final_index if final_index is not None else self._final_index_value
        if final is None or self.base_index_value is None:
            raise ValueError("Index values not set")
        return final / self.base_index_value

    def inflation_payment(self, final_index: Optional[float] = None) -> float:
        """Calculate inflation leg payment at maturity."""
        ratio = self.index_ratio(final_index)
        return self.notional * (ratio - 1.0)

    def fixed_payment(self) -> float:
        """Calculate fixed leg payment at maturity (compounded)."""
        # Compound fixed rate to maturity
        t = self.tenor_years
        compounded = (1 + self.fixed_rate.decimal) ** t - 1
        return self.notional * compounded

    def npv(
        self,
        discount_curve: Curve,
        inflation_curve: Optional["InflationCurve"] = None,
        valuation_date: Optional[date] = None
    ) -> float:
        """
        Calculate NPV.

        Args:
            discount_curve: Nominal discount curve
            inflation_curve: Inflation forecasting curve
            valuation_date: Valuation date

        Returns:
            NPV
        """
        df = discount_curve.discount_factor(self.termination_date)

        # Fixed leg NPV
        fixed_cf = self.fixed_payment()
        fixed_npv = -fixed_cf * df if self.pay_inflation else fixed_cf * df

        # Inflation leg NPV
        if self._final_index_value is not None:
            # Use actual final index
            inflation_cf = self.inflation_payment()
        elif inflation_curve is not None:
            # Forecast from inflation curve
            forecast_ratio = inflation_curve.index_ratio(
                self.effective_date,
                self.termination_date,
                self.base_index_value
            )
            inflation_cf = self.notional * (forecast_ratio - 1.0)
        else:
            # Cannot calculate
            raise ValueError("Need final index or inflation curve for NPV")

        inflation_npv = inflation_cf * df if self.pay_inflation else -inflation_cf * df

        return fixed_npv + inflation_npv

    def breakeven_rate(
        self,
        discount_curve: Curve,
        inflation_curve: "InflationCurve"
    ) -> float:
        """
        Calculate breakeven inflation rate.

        The fixed rate that makes NPV = 0.

        Returns:
            Breakeven rate as decimal
        """
        # Get forecast inflation ratio
        forecast_ratio = inflation_curve.index_ratio(
            self.effective_date,
            self.termination_date,
            self.base_index_value
        )

        # Breakeven: (1 + r)^T = forecast_ratio
        t = self.tenor_years
        if t > 0 and forecast_ratio > 0:
            return forecast_ratio ** (1 / t) - 1
        return 0.0

    def real_rate(
        self,
        nominal_rate: float
    ) -> float:
        """
        Calculate implied real rate (Fisher equation).

        real_rate = (1 + nominal) / (1 + inflation) - 1
        """
        return (1 + nominal_rate) / (1 + self.fixed_rate.decimal) - 1

    def inflation_dv01(self, discount_curve: Curve) -> float:
        """
        Calculate inflation DV01.

        Sensitivity to 1bp change in breakeven rate.
        """
        base_npv = self.npv_from_rate(discount_curve, self.fixed_rate.decimal)
        bumped_npv = self.npv_from_rate(discount_curve, self.fixed_rate.decimal + 0.0001)
        return bumped_npv - base_npv

    def npv_from_rate(
        self,
        discount_curve: Curve,
        breakeven_rate: float
    ) -> float:
        """Calculate NPV given a breakeven rate."""
        df = discount_curve.discount_factor(self.termination_date)
        t = self.tenor_years

        # Fixed leg: Notional * ((1 + r)^T - 1)
        fixed_cf = self.notional * ((1 + breakeven_rate) ** t - 1)
        fixed_npv = -fixed_cf * df if self.pay_inflation else fixed_cf * df

        # Inflation leg: assume breakeven is fair
        # At breakeven, NPV = 0, so inflation = fixed
        inflation_npv = -fixed_npv

        return fixed_npv + inflation_npv

    def __repr__(self) -> str:
        direction = "Pay" if self.pay_inflation else "Receive"
        return (
            f"ZeroCouponInflationSwap({direction} {self.inflation_index}, "
            f"BE={self.fixed_rate}, {self.tenor}, "
            f"notional={self.notional:,.0f} {self.currency.value})"
        )


# =============================================================================
# Year-on-Year Inflation Swap
# =============================================================================

@dataclass
class YearOnYearInflationSwap:
    """
    Year-on-Year Inflation Swap (YoY).

    Exchanges:
    - Fixed leg: Periodic fixed payments
    - Inflation leg: YoY inflation rate * Notional each period

    YoY inflation = (Index_t / Index_{t-1Y}) - 1

    Args:
        effective_date: Start date
        termination_date: End date
        notional: Notional amount
        fixed_rate: Fixed rate per period
        currency: Currency
        inflation_index: Inflation index name
        frequency: Payment frequency (typically "1Y")
        index_lag_months: Index lag
        pay_inflation: True = pay inflation, receive fixed
    """
    effective_date: date
    termination_date: date
    notional: float
    fixed_rate: Rate
    currency: Currency
    inflation_index: str = "CPI"
    frequency: str = "1Y"
    index_lag_months: int = 3
    pay_inflation: bool = True
    day_count: str = "ACT/ACT"

    _fixed_leg: Optional[CoreFixedLeg] = field(default=None, repr=False)
    _inflation_leg: Optional[InflationLeg] = field(default=None, repr=False)
    _yoy_rates: Dict[date, float] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.fixed_rate, (float, int, str)):
            self.fixed_rate = Rate(self.fixed_rate)
        if isinstance(self.currency, str):
            self.currency = Currency.from_string(self.currency)

        self._build_legs()

    def _build_legs(self) -> None:
        """Build fixed and inflation legs."""
        pay_receive_fixed = 1 if self.pay_inflation else -1
        pay_receive_infl = -1 if self.pay_inflation else 1

        self._fixed_leg = CoreFixedLeg(
            effective_date=self.effective_date,
            termination_date=self.termination_date,
            coupon_rate=self.fixed_rate,
            notional_amount=self.notional,
            ccy=self.currency,
            frequency=self.frequency,
            day_count=self.day_count,
            pay_receive=pay_receive_fixed
        )

        # For YoY, we need to track each year's index ratio
        self._inflation_leg = InflationLeg(
            effective_date=self.effective_date,
            termination_date=self.termination_date,
            base_index=100.0,  # Will be set per period
            notional_amount=self.notional,
            ccy=self.currency,
            index_name=self.inflation_index,
            frequency=self.frequency,
            index_lag_months=self.index_lag_months,
            pay_receive=pay_receive_infl
        )

    def set_yoy_rates(self, rates: Dict[date, float]) -> None:
        """Set observed YoY inflation rates by period end date."""
        self._yoy_rates = rates

    @property
    def tenor(self) -> str:
        """Swap tenor."""
        years = (self.termination_date - self.effective_date).days / 365.0
        return f"{int(round(years))}Y"

    def npv(
        self,
        discount_curve: Curve,
        inflation_curve: Optional["InflationCurve"] = None
    ) -> float:
        """Calculate NPV."""
        fixed_npv = self._fixed_leg.npv(discount_curve)

        # Inflation leg: sum of YoY payments
        inflation_npv = 0.0
        tenor = as_tenor(self.frequency)
        current = self.effective_date

        while current < self.termination_date:
            next_date = tenor.add_to_date(current)
            if next_date > self.termination_date:
                next_date = self.termination_date

            # Get YoY rate for this period
            if next_date in self._yoy_rates:
                yoy_rate = self._yoy_rates[next_date]
            elif inflation_curve is not None:
                # Forecast from curve
                yoy_rate = inflation_curve.yoy_rate(current, next_date)
            else:
                yoy_rate = 0.0

            # YoY payment
            dcf = (next_date - current).days / 365.0
            payment = self.notional * yoy_rate * dcf
            df = discount_curve.discount_factor(next_date)

            if self.pay_inflation:
                inflation_npv -= payment * df
            else:
                inflation_npv += payment * df

            current = next_date

        return fixed_npv + inflation_npv

    def par_rate(
        self,
        discount_curve: Curve,
        inflation_curve: "InflationCurve"
    ) -> float:
        """Calculate par fixed rate."""
        # Get fixed leg annuity
        annuity = self._fixed_leg.annuity(discount_curve)

        if abs(annuity) < 1e-10:
            return 0.0

        # Create zero-rate swap to get inflation leg NPV
        zero_swap = YearOnYearInflationSwap(
            effective_date=self.effective_date,
            termination_date=self.termination_date,
            notional=self.notional,
            fixed_rate=Rate(0.0),
            currency=self.currency,
            inflation_index=self.inflation_index,
            frequency=self.frequency,
            pay_inflation=self.pay_inflation
        )

        # At par rate, NPV = 0
        # fixed_leg_npv + inflation_leg_npv = 0
        # rate * annuity = inflation_leg_npv
        inflation_pv = zero_swap.npv(discount_curve, inflation_curve)

        return inflation_pv / annuity / self.notional

    def __repr__(self) -> str:
        direction = "Pay" if self.pay_inflation else "Receive"
        return (
            f"YearOnYearInflationSwap({direction} {self.inflation_index}, "
            f"fixed={self.fixed_rate}, {self.tenor})"
        )


# =============================================================================
# Inflation-Indexed Interest Rate Swap (IIRS)
# =============================================================================

@dataclass
class InflationIndexedSwap:
    """
    Inflation-Indexed Interest Rate Swap (IIRS).

    A swap where both notional and coupon are inflation-adjusted.

    Fixed leg: Fixed rate on inflation-adjusted notional
    Float leg: Floating rate on inflation-adjusted notional

    Used for:
    - Real rate trading
    - Inflation-linked bond hedging

    Args:
        effective_date: Start date
        termination_date: End date
        notional: Base notional amount
        fixed_rate: Fixed real rate
        currency: Currency
        inflation_index: Inflation index name
        base_index_value: Base index at start
        float_index: Floating rate index
        fixed_frequency: Fixed leg frequency
        float_frequency: Float leg frequency
    """
    effective_date: date
    termination_date: date
    notional: float
    fixed_rate: Rate
    currency: Currency
    inflation_index: str = "CPI"
    base_index_value: float = 100.0
    float_index: str = "SOFR"
    fixed_frequency: str = "6M"
    float_frequency: str = "3M"
    float_spread: Rate = field(default_factory=lambda: Rate(0.0))
    pay_fixed: bool = True
    day_count_fixed: str = "ACT/ACT"
    day_count_float: str = "ACT/360"

    _index_values: Dict[date, float] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.fixed_rate, (float, int, str)):
            self.fixed_rate = Rate(self.fixed_rate)
        if isinstance(self.float_spread, (float, int, str)):
            self.float_spread = Rate(self.float_spread)
        if isinstance(self.currency, str):
            self.currency = Currency.from_string(self.currency)

    def set_index_values(self, values: Dict[date, float]) -> None:
        """Set inflation index values by date."""
        self._index_values = values

    def inflation_factor(self, dt: date) -> float:
        """Calculate inflation factor at a date."""
        if dt in self._index_values:
            return self._index_values[dt] / self.base_index_value
        return 1.0

    def adjusted_notional(self, dt: date) -> float:
        """Get inflation-adjusted notional."""
        return self.notional * self.inflation_factor(dt)

    def npv(
        self,
        discount_curve: Curve,
        forecast_curve: Optional[Curve] = None,
        inflation_curve: Optional["InflationCurve"] = None
    ) -> float:
        """Calculate NPV with inflation adjustment."""
        fc = forecast_curve if forecast_curve is not None else discount_curve

        # Generate payment dates
        fixed_dates = self._generate_dates(self.fixed_frequency)
        float_dates = self._generate_dates(self.float_frequency)

        npv = 0.0

        # Fixed leg
        for i in range(len(fixed_dates) - 1):
            start = fixed_dates[i]
            end = fixed_dates[i + 1]
            dcf = (end - start).days / 365.0
            df = discount_curve.discount_factor(end)

            # Inflation-adjusted notional at payment date
            adj_notional = self.adjusted_notional(end)
            payment = adj_notional * self.fixed_rate.decimal * dcf

            if self.pay_fixed:
                npv -= payment * df
            else:
                npv += payment * df

        # Float leg
        for i in range(len(float_dates) - 1):
            start = float_dates[i]
            end = float_dates[i + 1]
            dcf = (end - start).days / 360.0
            df = discount_curve.discount_factor(end)

            # Forward rate
            fwd_rate = fc.forward_rate(start, end) + self.float_spread.decimal

            # Inflation-adjusted notional
            adj_notional = self.adjusted_notional(end)
            payment = adj_notional * fwd_rate * dcf

            if self.pay_fixed:
                npv += payment * df
            else:
                npv -= payment * df

        return npv

    def _generate_dates(self, frequency: str) -> List[date]:
        """Generate payment dates."""
        dates = [self.effective_date]
        tenor = as_tenor(frequency)
        current = self.effective_date

        while True:
            next_date = tenor.add_to_date(current)
            if next_date >= self.termination_date:
                dates.append(self.termination_date)
                break
            dates.append(next_date)
            current = next_date

        return dates

    def real_rate(self) -> float:
        """Get the real (inflation-adjusted) fixed rate."""
        return self.fixed_rate.decimal

    def __repr__(self) -> str:
        direction = "Pay" if self.pay_fixed else "Receive"
        tenor = f"{(self.termination_date - self.effective_date).days // 365}Y"
        return (
            f"InflationIndexedSwap({direction} {self.fixed_rate} real, "
            f"{self.inflation_index}, {tenor})"
        )
