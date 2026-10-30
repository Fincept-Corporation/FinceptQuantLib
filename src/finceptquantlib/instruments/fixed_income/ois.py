"""
Overnight Index Swaps (OIS)

Provides:
- OIS pricing and analytics
- OIS curve construction
- Fed Funds, SOFR, ESTR swaps
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, List
from enum import Enum
from finceptquantlib.curves.curve import Curve, DiscountCurve
from finceptquantlib.scheduling.daycount import DayCountConvention, Actual360


class OISIndex(Enum):
    """OIS reference index."""
    FED_FUNDS = "fed_funds"
    SOFR = "sofr"
    ESTR = "estr"
    SONIA = "sonia"
    TONA = "tona"


@dataclass
class OISResult:
    """OIS valuation result."""
    fixed_leg_pv: float
    float_leg_pv: float
    npv: float
    par_rate: float
    dv01: float
    accrued_interest: float


class OvernightIndexSwap:
    """
    Overnight Index Swap.

    Exchange of fixed rate for compounded overnight rate.
    """

    def __init__(
        self,
        notional: float,
        fixed_rate: float,
        start_date: date,
        maturity_date: date,
        pay_fixed: bool = True,
        index: OISIndex = OISIndex.SOFR,
        fixed_day_count: Optional[DayCountConvention] = None,
        float_day_count: Optional[DayCountConvention] = None,
        payment_lag: int = 2,
    ):
        self.notional = notional
        self.fixed_rate = fixed_rate
        self.start_date = start_date
        self.maturity_date = maturity_date
        self.pay_fixed = pay_fixed
        self.index = index
        self._fixed_dc = fixed_day_count or Actual360()
        self._float_dc = float_day_count or Actual360()
        self.payment_lag = payment_lag

    @classmethod
    def from_tenor(
        cls,
        notional: float,
        fixed_rate: float,
        value_date: date,
        tenor: str,
        **kwargs,
    ) -> "OvernightIndexSwap":
        """
        Create OIS from tenor string.

        Args:
            notional: Notional amount
            fixed_rate: Fixed rate
            value_date: Effective date
            tenor: Tenor string (e.g., "1W", "1M", "1Y")
        """
        # Parse tenor
        unit = tenor[-1].upper()
        amount = int(tenor[:-1])

        if unit == "D":
            maturity = value_date + timedelta(days=amount)
        elif unit == "W":
            maturity = value_date + timedelta(weeks=amount)
        elif unit == "M":
            maturity = value_date + timedelta(days=amount * 30)
        elif unit == "Y":
            maturity = value_date + timedelta(days=amount * 365)
        else:
            raise ValueError(f"Unknown tenor unit: {unit}")

        return cls(
            notional=notional,
            fixed_rate=fixed_rate,
            start_date=value_date,
            maturity_date=maturity,
            **kwargs,
        )

    def year_fraction(self) -> float:
        """Calculate accrual year fraction."""
        return self._fixed_dc.year_fraction(self.start_date, self.maturity_date)

    def fixed_leg_pv(
        self,
        valuation_date: date,
        discount_curve: Curve,
    ) -> float:
        """
        Calculate fixed leg present value.

        For OIS, typically single payment at maturity.
        """
        if valuation_date >= self.maturity_date:
            return 0.0

        t = (self.maturity_date - valuation_date).days / 365.0
        df = discount_curve.discount_factor(t)

        yf = self.year_fraction()
        fixed_payment = self.notional * self.fixed_rate * yf

        return fixed_payment * df

    def float_leg_pv(
        self,
        valuation_date: date,
        ois_curve: Curve,
        discount_curve: Optional[Curve] = None,
    ) -> float:
        """
        Calculate floating leg present value.

        Floating leg pays compounded overnight rate.
        PV = N * (DF(start)/DF(end) - 1) * DF(end)
            = N * (DF(start) - DF(end))
        """
        if discount_curve is None:
            discount_curve = ois_curve

        if valuation_date >= self.maturity_date:
            return 0.0

        t_start = max(0, (self.start_date - valuation_date).days / 365.0)
        t_end = (self.maturity_date - valuation_date).days / 365.0

        df_start = ois_curve.discount_factor(t_start)
        df_end = ois_curve.discount_factor(t_end)

        # Floating leg PV
        float_pv = self.notional * (df_start - df_end)

        # If different discount curve, adjust
        if discount_curve is not ois_curve:
            disc_df_end = discount_curve.discount_factor(t_end)
            float_pv = self.notional * (df_start / df_end - 1) * disc_df_end

        return float_pv

    def present_value(
        self,
        valuation_date: date,
        ois_curve: Curve,
        discount_curve: Optional[Curve] = None,
    ) -> float:
        """
        Calculate swap NPV.

        Args:
            valuation_date: Valuation date
            ois_curve: OIS projection curve
            discount_curve: Discount curve (if different from OIS)

        Returns:
            Net present value
        """
        fixed_pv = self.fixed_leg_pv(valuation_date, discount_curve or ois_curve)
        float_pv = self.float_leg_pv(valuation_date, ois_curve, discount_curve)

        if self.pay_fixed:
            return float_pv - fixed_pv
        else:
            return fixed_pv - float_pv

    def par_rate(
        self,
        valuation_date: date,
        ois_curve: Curve,
        discount_curve: Optional[Curve] = None,
    ) -> float:
        """
        Calculate par OIS rate.

        Rate that makes NPV = 0.
        """
        if discount_curve is None:
            discount_curve = ois_curve

        t_start = max(0, (self.start_date - valuation_date).days / 365.0)
        t_end = (self.maturity_date - valuation_date).days / 365.0

        df_start = ois_curve.discount_factor(t_start)
        df_end = ois_curve.discount_factor(t_end)

        yf = self.year_fraction()

        # Par rate (independent of the discount curve: both legs pay at maturity) = (DF_start/DF_end - 1) / (yf)
        return (df_start / df_end - 1) / yf

    def dv01(
        self,
        valuation_date: date,
        ois_curve: Curve,
        discount_curve: Optional[Curve] = None,
        bump_size: float = 0.0001,
    ) -> float:
        """Calculate DV01 (sensitivity to 1bp rate move)."""
        base_pv = self.present_value(valuation_date, ois_curve, discount_curve)

        # Create bumped swap
        bumped_swap = OvernightIndexSwap(
            notional=self.notional,
            fixed_rate=self.fixed_rate + bump_size,
            start_date=self.start_date,
            maturity_date=self.maturity_date,
            pay_fixed=self.pay_fixed,
            index=self.index,
        )

        bumped_pv = bumped_swap.present_value(valuation_date, ois_curve, discount_curve)
        return (bumped_pv - base_pv) / bump_size

    def forward_rate(
        self,
        valuation_date: date,
        ois_curve: Curve,
    ) -> float:
        """Calculate forward OIS rate."""
        return self.par_rate(valuation_date, ois_curve)


class BasisSwap:
    """
    Basis Swap (tenor swap).

    Exchange of two floating rates with different tenors.
    E.g., 3M LIBOR vs 6M LIBOR + spread.
    """

    def __init__(
        self,
        notional: float,
        spread: float,
        start_date: date,
        maturity_date: date,
        pay_tenor: str = "3M",
        receive_tenor: str = "6M",
        spread_on_pay: bool = False,
    ):
        self.notional = notional
        self.spread = spread
        self.start_date = start_date
        self.maturity_date = maturity_date
        self.pay_tenor = pay_tenor
        self.receive_tenor = receive_tenor
        self.spread_on_pay = spread_on_pay

    def _tenor_to_months(self, tenor: str) -> int:
        """Convert tenor string to months."""
        unit = tenor[-1].upper()
        amount = int(tenor[:-1])
        if unit == "M":
            return amount
        elif unit == "Y":
            return amount * 12
        elif unit == "W":
            return 0  # Less than a month
        return amount

    def pay_leg_pv(
        self,
        valuation_date: date,
        pay_curve: Curve,
        discount_curve: Curve,
    ) -> float:
        """Calculate pay leg present value."""
        # Simplified: single period
        t_end = (self.maturity_date - valuation_date).days / 365.0
        if t_end <= 0:
            return 0.0

        t_start = max(0, (self.start_date - valuation_date).days / 365.0)

        df_start = pay_curve.discount_factor(t_start)
        df_end = pay_curve.discount_factor(t_end)

        # Forward rate
        yf = t_end - t_start
        fwd_rate = (df_start / df_end - 1) / yf

        # Add spread if on pay leg
        if self.spread_on_pay:
            fwd_rate += self.spread

        disc_df = discount_curve.discount_factor(t_end)
        return self.notional * fwd_rate * yf * disc_df

    def receive_leg_pv(
        self,
        valuation_date: date,
        receive_curve: Curve,
        discount_curve: Curve,
    ) -> float:
        """Calculate receive leg present value."""
        t_end = (self.maturity_date - valuation_date).days / 365.0
        if t_end <= 0:
            return 0.0

        t_start = max(0, (self.start_date - valuation_date).days / 365.0)

        df_start = receive_curve.discount_factor(t_start)
        df_end = receive_curve.discount_factor(t_end)

        yf = t_end - t_start
        fwd_rate = (df_start / df_end - 1) / yf

        # Add spread if on receive leg
        if not self.spread_on_pay:
            fwd_rate += self.spread

        disc_df = discount_curve.discount_factor(t_end)
        return self.notional * fwd_rate * yf * disc_df

    def present_value(
        self,
        valuation_date: date,
        pay_curve: Curve,
        receive_curve: Curve,
        discount_curve: Curve,
    ) -> float:
        """Calculate basis swap NPV."""
        pay_pv = self.pay_leg_pv(valuation_date, pay_curve, discount_curve)
        rec_pv = self.receive_leg_pv(valuation_date, receive_curve, discount_curve)
        return rec_pv - pay_pv

    def par_spread(
        self,
        valuation_date: date,
        pay_curve: Curve,
        receive_curve: Curve,
        discount_curve: Curve,
    ) -> float:
        """
        Calculate par basis spread.

        Spread that makes NPV = 0.
        """
        # Save original spread
        original_spread = self.spread

        # Calculate PV with zero spread
        self.spread = 0.0
        zero_spread_pv = self.present_value(
            valuation_date, pay_curve, receive_curve, discount_curve
        )

        # Calculate annuity
        t_end = (self.maturity_date - valuation_date).days / 365.0
        t_start = max(0, (self.start_date - valuation_date).days / 365.0)
        yf = t_end - t_start
        disc_df = discount_curve.discount_factor(t_end)
        annuity = self.notional * yf * disc_df

        # Restore spread
        self.spread = original_spread

        # Par spread = -zero_spread_pv / annuity
        if abs(annuity) < 1e-10:
            return 0.0

        return -zero_spread_pv / annuity


def build_ois_curve(
    reference_date: date,
    ois_rates: List[tuple],  # [(tenor, rate), ...]
) -> DiscountCurve:
    """
    Build OIS discount curve from market rates.

    Args:
        reference_date: Curve reference date
        ois_rates: List of (tenor_string, rate) tuples

    Returns:
        Bootstrapped discount curve
    """
    tenors = [0.0]
    dfs = [1.0]

    for tenor_str, rate in sorted(ois_rates, key=lambda x: _tenor_to_years(x[0])):
        t = _tenor_to_years(tenor_str)
        # OIS: DF = 1 / (1 + rate * t) for simple compounding
        df = 1.0 / (1.0 + rate * t)
        tenors.append(t)
        dfs.append(df)

    return DiscountCurve(
        reference_date=reference_date,
        tenors=tenors,
        values=dfs,
        interpolation="log_linear",
    )


def _tenor_to_years(tenor: str) -> float:
    """Convert tenor string to years."""
    unit = tenor[-1].upper()
    amount = int(tenor[:-1])

    if unit == "D":
        return amount / 365.0
    elif unit == "W":
        return amount * 7 / 365.0
    elif unit == "M":
        return amount / 12.0
    elif unit == "Y":
        return float(amount)
    else:
        raise ValueError(f"Unknown tenor unit: {unit}")
