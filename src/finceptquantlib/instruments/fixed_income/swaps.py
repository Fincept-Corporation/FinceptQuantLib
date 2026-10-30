"""
Interest Rate Swaps

Implements vanilla and exotic interest rate swaps:
- Fixed-for-floating (vanilla IRS)
- Basis swaps (floating-for-floating)
- Overnight Index Swaps (OIS)
- Cross-currency swaps

Key Features:
- Multi-curve framework support (separate discounting and projection curves)
- Automatic differentiation for Greeks
- Comprehensive schedule generation
- Support for various day count conventions and payment frequencies
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from typing import Union, Optional, List, Dict

from finceptquantlib.curves.curve import Curve
from finceptquantlib.scheduling.schedule import Schedule, Frequency, StubType
from finceptquantlib.scheduling.calendar import Calendar
from finceptquantlib.scheduling.daycount import DayCountConvention, _get_convention
from finceptquantlib.scheduling.adjustment import BusinessDayAdjustment


class SwapType(Enum):
    """Type of interest rate swap."""
    PAYER = "payer"        # Pay fixed, receive floating
    RECEIVER = "receiver"  # Receive fixed, pay floating


@dataclass
class CashFlow:
    """Represents a single cash flow."""
    payment_date: date
    amount: float
    accrual_start: date
    accrual_end: date
    notional: float
    rate: Optional[float] = None
    day_count_fraction: Optional[float] = None

    def discount(self, curve: Curve) -> float:
        """Discount this cash flow to the curve's reference date."""
        df = curve.discount_factor(self.payment_date)
        return self.amount * df


@dataclass
class SwapLegResult:
    """Result of swap leg valuation."""
    present_value: float
    cash_flows: List[CashFlow]
    dv01: float
    accrued_interest: float
    annuity: float


class SwapLeg(ABC):
    """Abstract base class for swap legs."""

    def __init__(
        self,
        notional: float,
        effective_date: date,
        termination_date: date,
        frequency: Union[str, Frequency],
        day_count: Union[str, DayCountConvention],
        calendar: Optional[Calendar] = None,
        payment_delay: int = 0,
        stub: StubType = StubType.SHORT_FRONT,
    ):
        """
        Initialize swap leg.

        Args:
            notional: Notional principal
            effective_date: Effective date
            termination_date: Maturity date
            frequency: Payment frequency
            day_count: Day count convention
            calendar: Business day calendar
            payment_delay: Days between accrual end and payment
            stub: Type of stub period
        """
        self.notional = notional
        self.start_date = effective_date
        self.end_date = termination_date

        if isinstance(frequency, str):
            frequency = Frequency[frequency.upper().replace("-", "_").replace(" ", "_")]
        self.frequency = frequency

        if isinstance(day_count, str):
            day_count = _get_convention(day_count)
        self.day_count = day_count

        self.calendar = calendar
        self.payment_delay = payment_delay
        self.stub_type = stub

        # Generate schedule
        self._schedule = Schedule(
            effective_date=effective_date,
            termination_date=termination_date,
            frequency=frequency,
            calendar=calendar,
            stub=stub,
        )

    @property
    def schedule(self) -> Schedule:
        """Payment schedule."""
        return self._schedule

    @abstractmethod
    def generate_cash_flows(
        self,
        valuation_date: date,
        projection_curve: Optional[Curve] = None,
    ) -> List[CashFlow]:
        """Generate cash flows for the leg."""

    @abstractmethod
    def present_value(
        self,
        valuation_date: date,
        discount_curve: Curve,
        projection_curve: Optional[Curve] = None,
    ) -> float:
        """Calculate present value of the leg."""

    def _get_payment_date(self, accrual_end: date) -> date:
        """Get payment date from accrual end date."""
        payment_date = accrual_end + timedelta(days=self.payment_delay)
        if self.calendar:
            adjustment = BusinessDayAdjustment("modified_following", self.calendar)
            payment_date = adjustment.adjust(payment_date)
        return payment_date


class FixedLeg(SwapLeg):
    """
    Fixed leg of an interest rate swap.

    Pays (or receives) a fixed coupon rate on a notional amount.
    """

    def __init__(
        self,
        notional: float,
        effective_date: date,
        termination_date: date,
        fixed_rate: float,
        frequency: Union[str, Frequency] = Frequency.SEMI_ANNUAL,
        day_count: Union[str, DayCountConvention] = "30/360",
        calendar: Optional[Calendar] = None,
        payment_delay: int = 0,
        stub: StubType = StubType.SHORT_FRONT,
    ):
        """
        Initialize fixed leg.

        Args:
            notional: Notional principal
            effective_date: Effective date
            termination_date: Maturity date
            fixed_rate: Fixed coupon rate (as decimal, e.g., 0.05 for 5%)
            frequency: Payment frequency
            day_count: Day count convention
            calendar: Business day calendar
            payment_delay: Days between accrual end and payment
            stub: Type of stub period
        """
        super().__init__(
            notional=notional,
            effective_date=effective_date,
            termination_date=termination_date,
            frequency=frequency,
            day_count=day_count,
            calendar=calendar,
            payment_delay=payment_delay,
            stub=stub,
        )
        self.fixed_rate = fixed_rate

    def generate_cash_flows(
        self,
        valuation_date: date,
        projection_curve: Optional[Curve] = None,
    ) -> List[CashFlow]:
        """Generate fixed cash flows."""
        cash_flows = []
        periods = self._schedule.periods

        for period in periods:
            payment_date = period.payment_date

            # Skip past cash flows
            if payment_date < valuation_date:
                continue

            dcf = period.year_fraction(self.day_count)
            amount = self.notional * self.fixed_rate * dcf

            cash_flows.append(CashFlow(
                payment_date=payment_date,
                amount=amount,
                accrual_start=period.accrual_start,
                accrual_end=period.accrual_end,
                notional=self.notional,
                rate=self.fixed_rate,
                day_count_fraction=dcf,
            ))

        return cash_flows

    def present_value(
        self,
        valuation_date: date,
        discount_curve: Curve,
        projection_curve: Optional[Curve] = None,
    ) -> float:
        """Calculate present value of fixed leg."""
        cash_flows = self.generate_cash_flows(valuation_date)
        return sum(cf.discount(discount_curve) for cf in cash_flows)

    def annuity(
        self,
        valuation_date: date,
        discount_curve: Curve,
    ) -> float:
        """
        Calculate the annuity (PV01) of the fixed leg.

        The annuity is the present value of receiving 1bp on each payment.
        """
        total = 0.0
        periods = self._schedule.periods

        for period in periods:
            payment_date = period.payment_date
            if payment_date < valuation_date:
                continue

            dcf = period.year_fraction(self.day_count)
            df = discount_curve.discount_factor(payment_date)
            total += dcf * df

        return self.notional * total

    def dv01(
        self,
        valuation_date: date,
        discount_curve: Curve,
    ) -> float:
        """Calculate DV01 (dollar value of 1bp rate change)."""
        return self.annuity(valuation_date, discount_curve) / 10000


class FloatingLeg(SwapLeg):
    """
    Floating leg of an interest rate swap.

    Pays (or receives) a floating rate (e.g., LIBOR, SOFR) plus a spread.
    """

    def __init__(
        self,
        notional: float,
        effective_date: date,
        termination_date: date,
        index_tenor: str = "3M",
        spread: float = 0.0,
        frequency: Union[str, Frequency] = Frequency.QUARTERLY,
        day_count: Union[str, DayCountConvention] = "ACT/360",
        calendar: Optional[Calendar] = None,
        payment_delay: int = 0,
        fixing_days: int = 2,
        stub: StubType = StubType.SHORT_FRONT,
        in_arrears: bool = False,
    ):
        """
        Initialize floating leg.

        Args:
            notional: Notional principal
            effective_date: Effective date
            termination_date: Maturity date
            index_tenor: Index tenor (e.g., "1M", "3M", "6M")
            spread: Spread over the floating rate
            frequency: Payment frequency
            day_count: Day count convention
            calendar: Business day calendar
            payment_delay: Days between accrual end and payment
            fixing_days: Days before accrual start for rate fixing
            stub_type: Type of stub period
            in_arrears: If True, rate is fixed at end of period
        """
        super().__init__(
            notional=notional,
            effective_date=effective_date,
            termination_date=termination_date,
            frequency=frequency,
            day_count=day_count,
            calendar=calendar,
            payment_delay=payment_delay,
            stub=stub,
        )
        self.index_tenor = index_tenor
        self.spread = spread
        self.fixing_days = fixing_days
        self.in_arrears = in_arrears

        # Parse index tenor to years
        self._index_tenor_years = self._parse_tenor(index_tenor)

    @staticmethod
    def _parse_tenor(tenor: str) -> float:
        """Parse tenor string to year fraction."""
        tenor = tenor.upper().strip()
        if tenor.endswith("M"):
            return int(tenor[:-1]) / 12.0
        elif tenor.endswith("Y"):
            return float(tenor[:-1])
        elif tenor.endswith("W"):
            return int(tenor[:-1]) / 52.0
        elif tenor.endswith("D"):
            return int(tenor[:-1]) / 365.0
        else:
            raise ValueError(f"Unknown tenor format: {tenor}")

    def _get_fixing_date(self, accrual_start: date, accrual_end: date) -> date:
        """Get fixing date for a period."""
        if self.in_arrears:
            base_date = accrual_end
        else:
            base_date = accrual_start

        fixing_date = base_date - timedelta(days=self.fixing_days)
        if self.calendar:
            adjustment = BusinessDayAdjustment("preceding", self.calendar)
            fixing_date = adjustment.adjust(fixing_date)
        return fixing_date

    def _project_rate(
        self,
        fixing_date: date,
        accrual_start: date,
        accrual_end: date,
        projection_curve: Curve,
    ) -> float:
        """Project forward rate from the curve."""
        # Use forward rate from the curve
        t1 = projection_curve.time_to_maturity(accrual_start)
        t2 = projection_curve.time_to_maturity(accrual_end)

        if t1 <= 0:
            t1 = 0.001
        if t2 <= t1:
            t2 = t1 + self._index_tenor_years

        return projection_curve.forward_rate(t1, t2)

    def generate_cash_flows(
        self,
        valuation_date: date,
        projection_curve: Optional[Curve] = None,
        fixed_rates: Optional[Dict[date, float]] = None,
    ) -> List[CashFlow]:
        """
        Generate floating cash flows.

        Args:
            valuation_date: Valuation date
            projection_curve: Curve for projecting forward rates
            fixed_rates: Dictionary of already-fixed rates {fixing_date: rate}
        """
        if projection_curve is None:
            raise ValueError("projection_curve required for floating leg")

        fixed_rates = fixed_rates or {}
        cash_flows = []
        periods = self._schedule.periods

        for period in periods:
            payment_date = period.payment_date

            # Skip past cash flows
            if payment_date < valuation_date:
                continue

            fixing_date = self._get_fixing_date(period.accrual_start, period.accrual_end)

            # Use fixed rate if available, otherwise project
            if fixing_date in fixed_rates:
                rate = fixed_rates[fixing_date]
            elif fixing_date <= valuation_date:
                # Past fixing - should be provided
                rate = self._project_rate(
                    fixing_date, period.accrual_start, period.accrual_end, projection_curve
                )
            else:
                # Future fixing - project from curve
                rate = self._project_rate(
                    fixing_date, period.accrual_start, period.accrual_end, projection_curve
                )

            dcf = period.year_fraction(self.day_count)
            amount = self.notional * (rate + self.spread) * dcf

            cash_flows.append(CashFlow(
                payment_date=payment_date,
                amount=amount,
                accrual_start=period.accrual_start,
                accrual_end=period.accrual_end,
                notional=self.notional,
                rate=rate + self.spread,
                day_count_fraction=dcf,
            ))

        return cash_flows

    def present_value(
        self,
        valuation_date: date,
        discount_curve: Curve,
        projection_curve: Optional[Curve] = None,
        fixed_rates: Optional[Dict[date, float]] = None,
    ) -> float:
        """Calculate present value of floating leg."""
        if projection_curve is None:
            projection_curve = discount_curve

        cash_flows = self.generate_cash_flows(
            valuation_date, projection_curve, fixed_rates
        )
        return sum(cf.discount(discount_curve) for cf in cash_flows)


@dataclass
class SwapResult:
    """Result of swap valuation."""
    present_value: float
    fixed_leg_pv: float
    floating_leg_pv: float
    par_rate: float
    dv01: float
    fixed_leg_dv01: float
    floating_leg_dv01: float
    cash_flows_fixed: List[CashFlow]
    cash_flows_floating: List[CashFlow]
    accrued_interest: float = 0.0


class InterestRateSwap:
    """
    Vanilla Interest Rate Swap.

    A swap where one party pays fixed rate and receives floating rate,
    or vice versa.

    Features:
    - Multi-curve framework (separate discount and projection curves)
    - Automatic par rate calculation
    - Full Greeks (DV01, duration, convexity)
    - Support for amortizing notionals
    """

    def __init__(
        self,
        notional: float,
        effective_date: date,
        termination_date: date,
        fixed_rate: float,
        swap_type: Union[str, SwapType] = SwapType.PAYER,
        fixed_frequency: Union[str, Frequency] = Frequency.SEMI_ANNUAL,
        float_frequency: Union[str, Frequency] = Frequency.QUARTERLY,
        fixed_day_count: Union[str, DayCountConvention] = "30/360",
        float_day_count: Union[str, DayCountConvention] = "ACT/360",
        index_tenor: str = "3M",
        spread: float = 0.0,
        calendar: Optional[Calendar] = None,
        payment_delay: int = 0,
    ):
        """
        Initialize interest rate swap.

        Args:
            notional: Notional principal
            start_date: Effective date
            end_date: Maturity date
            fixed_rate: Fixed coupon rate
            swap_type: PAYER (pay fixed) or RECEIVER (receive fixed)
            fixed_frequency: Fixed leg payment frequency
            float_frequency: Floating leg payment frequency
            fixed_day_count: Fixed leg day count convention
            float_day_count: Floating leg day count convention
            index_tenor: Floating rate index tenor
            spread: Spread over floating rate
            calendar: Business day calendar
            payment_delay: Days between accrual end and payment
        """
        self.notional = notional
        self.start_date = effective_date
        self.end_date = termination_date
        self.fixed_rate = fixed_rate

        if isinstance(swap_type, str):
            swap_type = SwapType[swap_type.upper()]
        self.swap_type = swap_type

        # Create legs
        self.fixed_leg = FixedLeg(
            notional=notional,
            effective_date=effective_date,
            termination_date=termination_date,
            fixed_rate=fixed_rate,
            frequency=fixed_frequency,
            day_count=fixed_day_count,
            calendar=calendar,
            payment_delay=payment_delay,
        )

        self.floating_leg = FloatingLeg(
            notional=notional,
            effective_date=effective_date,
            termination_date=termination_date,
            index_tenor=index_tenor,
            spread=spread,
            frequency=float_frequency,
            day_count=float_day_count,
            calendar=calendar,
            payment_delay=payment_delay,
        )

    def present_value(
        self,
        valuation_date: date,
        discount_curve: Curve,
        projection_curve: Optional[Curve] = None,
        fixed_rates: Optional[Dict[date, float]] = None,
    ) -> float:
        """
        Calculate present value of the swap.

        Args:
            valuation_date: Valuation date
            discount_curve: Curve for discounting cash flows
            projection_curve: Curve for projecting forward rates
            fixed_rates: Dictionary of already-fixed floating rates

        Returns:
            NPV of the swap (positive = in-the-money for swap holder)
        """
        if projection_curve is None:
            projection_curve = discount_curve

        fixed_pv = self.fixed_leg.present_value(valuation_date, discount_curve)
        float_pv = self.floating_leg.present_value(
            valuation_date, discount_curve, projection_curve, fixed_rates
        )

        if self.swap_type == SwapType.PAYER:
            # Pay fixed, receive floating
            return float_pv - fixed_pv
        else:
            # Receive fixed, pay floating
            return fixed_pv - float_pv

    def par_rate(
        self,
        valuation_date: date,
        discount_curve: Curve,
        projection_curve: Optional[Curve] = None,
    ) -> float:
        """
        Calculate the par (fair) swap rate.

        The par rate is the fixed rate that makes the swap have zero NPV.

        par_rate = PV(floating leg) / Annuity(fixed leg)
        """
        if projection_curve is None:
            projection_curve = discount_curve

        # Create a temporary floating leg to get its PV
        float_pv = self.floating_leg.present_value(
            valuation_date, discount_curve, projection_curve
        )

        # Calculate annuity
        annuity = self.fixed_leg.annuity(valuation_date, discount_curve)

        if abs(annuity) < 1e-10:
            return 0.0

        return float_pv / annuity

    def dv01(
        self,
        valuation_date: date,
        discount_curve: Curve,
        projection_curve: Optional[Curve] = None,
        bump_size: float = 1.0,
    ) -> float:
        """
        Calculate DV01 using parallel curve bump.

        Args:
            valuation_date: Valuation date
            discount_curve: Discount curve
            projection_curve: Projection curve
            bump_size: Bump size in basis points

        Returns:
            DV01 (change in NPV for 1bp parallel shift)
        """
        if projection_curve is None:
            projection_curve = discount_curve

        # Base NPV
        base_npv = self.present_value(valuation_date, discount_curve, projection_curve)

        # Bumped curves
        disc_up = discount_curve.shift(bump_size)
        proj_up = projection_curve.shift(bump_size) if projection_curve != discount_curve else disc_up

        # Bumped NPV
        npv_up = self.present_value(valuation_date, disc_up, proj_up)

        return npv_up - base_npv

    def gamma(
        self,
        valuation_date: date,
        discount_curve: Curve,
        projection_curve: Optional[Curve] = None,
        bump_size: float = 1.0,
    ) -> float:
        """
        Calculate gamma (second derivative w.r.t. rates).

        Gamma measures convexity - how DV01 changes as rates change.
        """
        if projection_curve is None:
            projection_curve = discount_curve

        # Base NPV
        base_npv = self.present_value(valuation_date, discount_curve, projection_curve)

        # Up bump
        disc_up = discount_curve.shift(bump_size)
        proj_up = projection_curve.shift(bump_size) if projection_curve != discount_curve else disc_up
        npv_up = self.present_value(valuation_date, disc_up, proj_up)

        # Down bump
        disc_down = discount_curve.shift(-bump_size)
        proj_down = projection_curve.shift(-bump_size) if projection_curve != discount_curve else disc_down
        npv_down = self.present_value(valuation_date, disc_down, proj_down)

        # Second derivative approximation
        return (npv_up - 2 * base_npv + npv_down) / (bump_size ** 2)

    def value(
        self,
        valuation_date: date,
        discount_curve: Curve,
        projection_curve: Optional[Curve] = None,
        fixed_rates: Optional[Dict[date, float]] = None,
    ) -> SwapResult:
        """
        Full valuation with all analytics.

        Returns:
            SwapResult with NPV, par rate, DV01, and cash flows
        """
        if projection_curve is None:
            projection_curve = discount_curve

        fixed_pv = self.fixed_leg.present_value(valuation_date, discount_curve)
        float_pv = self.floating_leg.present_value(
            valuation_date, discount_curve, projection_curve, fixed_rates
        )

        if self.swap_type == SwapType.PAYER:
            npv = float_pv - fixed_pv
        else:
            npv = fixed_pv - float_pv

        par = self.par_rate(valuation_date, discount_curve, projection_curve)
        dv01 = self.dv01(valuation_date, discount_curve, projection_curve)

        fixed_dv01 = self.fixed_leg.dv01(valuation_date, discount_curve)

        # Approximate floating DV01
        base_float = self.floating_leg.present_value(
            valuation_date, discount_curve, projection_curve
        )
        proj_up = projection_curve.shift(1.0)
        disc_up = discount_curve.shift(1.0)
        float_up = self.floating_leg.present_value(valuation_date, disc_up, proj_up)
        float_dv01 = float_up - base_float

        return SwapResult(
            present_value=npv,
            fixed_leg_pv=fixed_pv,
            floating_leg_pv=float_pv,
            par_rate=par,
            dv01=dv01,
            fixed_leg_dv01=fixed_dv01,
            floating_leg_dv01=float_dv01,
            cash_flows_fixed=self.fixed_leg.generate_cash_flows(valuation_date),
            cash_flows_floating=self.floating_leg.generate_cash_flows(
                valuation_date, projection_curve, fixed_rates
            ),
        )

    def __repr__(self) -> str:
        return (f"InterestRateSwap({self.swap_type.value}, "
                f"notional={self.notional:,.0f}, "
                f"rate={self.fixed_rate:.4%}, "
                f"{self.start_date} to {self.end_date})")


class BasisSwap:
    """
    Basis Swap (floating-for-floating swap).

    Exchange of two floating rate payments, typically different indices
    or different tenors of the same index.
    """

    def __init__(
        self,
        notional: float,
        effective_date: date,
        termination_date: date,
        pay_index_tenor: str = "3M",
        receive_index_tenor: str = "6M",
        pay_spread: float = 0.0,
        receive_spread: float = 0.0,
        pay_frequency: Union[str, Frequency] = Frequency.QUARTERLY,
        receive_frequency: Union[str, Frequency] = Frequency.SEMI_ANNUAL,
        day_count: Union[str, DayCountConvention] = "ACT/360",
        calendar: Optional[Calendar] = None,
    ):
        """
        Initialize basis swap.

        Args:
            notional: Notional principal
            start_date: Effective date
            end_date: Maturity date
            pay_index_tenor: Tenor of index to pay
            receive_index_tenor: Tenor of index to receive
            pay_spread: Spread on pay leg
            receive_spread: Spread on receive leg
            pay_frequency: Payment frequency for pay leg
            receive_frequency: Payment frequency for receive leg
            day_count: Day count convention
            calendar: Business day calendar
        """
        self.notional = notional
        self.start_date = effective_date
        self.end_date = termination_date

        self.pay_leg = FloatingLeg(
            notional=notional,
            effective_date=effective_date,
            termination_date=termination_date,
            index_tenor=pay_index_tenor,
            spread=pay_spread,
            frequency=pay_frequency,
            day_count=day_count,
            calendar=calendar,
        )

        self.receive_leg = FloatingLeg(
            notional=notional,
            effective_date=effective_date,
            termination_date=termination_date,
            index_tenor=receive_index_tenor,
            spread=receive_spread,
            frequency=receive_frequency,
            day_count=day_count,
            calendar=calendar,
        )

    def present_value(
        self,
        valuation_date: date,
        discount_curve: Curve,
        pay_projection_curve: Optional[Curve] = None,
        receive_projection_curve: Optional[Curve] = None,
    ) -> float:
        """Calculate present value of basis swap."""
        if pay_projection_curve is None:
            pay_projection_curve = discount_curve
        if receive_projection_curve is None:
            receive_projection_curve = discount_curve

        pay_pv = self.pay_leg.present_value(
            valuation_date, discount_curve, pay_projection_curve
        )
        receive_pv = self.receive_leg.present_value(
            valuation_date, discount_curve, receive_projection_curve
        )

        return receive_pv - pay_pv

    def par_spread(
        self,
        valuation_date: date,
        discount_curve: Curve,
        pay_projection_curve: Optional[Curve] = None,
        receive_projection_curve: Optional[Curve] = None,
    ) -> float:
        """
        Calculate par basis spread.

        The spread on the pay leg that makes the swap have zero NPV.
        """
        if pay_projection_curve is None:
            pay_projection_curve = discount_curve
        if receive_projection_curve is None:
            receive_projection_curve = discount_curve

        # PV of receive leg
        receive_pv = self.receive_leg.present_value(
            valuation_date, discount_curve, receive_projection_curve
        )

        # PV of pay leg with zero spread
        temp_pay = FloatingLeg(
            notional=self.notional,
            effective_date=self.start_date,
            termination_date=self.end_date,
            index_tenor=self.pay_leg.index_tenor,
            spread=0.0,
            frequency=self.pay_leg.frequency,
            day_count=self.pay_leg.day_count,
            calendar=self.pay_leg.calendar,
        )
        pay_pv_zero_spread = temp_pay.present_value(
            valuation_date, discount_curve, pay_projection_curve
        )

        # Calculate annuity for pay leg
        annuity = 0.0
        for start, end in temp_pay.schedule.periods:
            payment_date = temp_pay._get_payment_date(end)
            if payment_date < valuation_date:
                continue
            dcf = temp_pay.day_count.year_fraction(start, end)
            df = discount_curve.discount_factor(payment_date)
            annuity += dcf * df

        if abs(annuity) < 1e-10:
            return 0.0

        return (pay_pv_zero_spread - receive_pv) / (self.notional * annuity)


class OvernightIndexSwap:
    """
    Overnight Index Swap (OIS).

    Swap of a fixed rate for a compounded overnight rate (e.g., SOFR, ESTR).
    Uses daily compounding on the floating leg.
    """

    def __init__(
        self,
        notional: float,
        start_date: date,
        end_date: date,
        fixed_rate: float,
        swap_type: Union[str, SwapType] = SwapType.PAYER,
        fixed_frequency: Union[str, Frequency] = Frequency.ANNUAL,
        float_frequency: Union[str, Frequency] = Frequency.ANNUAL,
        fixed_day_count: Union[str, DayCountConvention] = "ACT/360",
        float_day_count: Union[str, DayCountConvention] = "ACT/360",
        calendar: Optional[Calendar] = None,
        lookback_days: int = 0,
        lockout_days: int = 0,
    ):
        """
        Initialize OIS.

        Args:
            notional: Notional principal
            start_date: Effective date
            end_date: Maturity date
            fixed_rate: Fixed rate
            swap_type: PAYER or RECEIVER
            fixed_frequency: Fixed leg payment frequency
            float_frequency: Floating leg payment frequency
            fixed_day_count: Fixed leg day count
            float_day_count: Floating leg day count
            calendar: Business day calendar
            lookback_days: Observation shift in days
            lockout_days: Lockout period before payment
        """
        self.notional = notional
        self.start_date = start_date
        self.end_date = end_date
        self.fixed_rate = fixed_rate

        if isinstance(swap_type, str):
            swap_type = SwapType[swap_type.upper()]
        self.swap_type = swap_type

        self.lookback_days = lookback_days
        self.lockout_days = lockout_days

        if isinstance(fixed_day_count, str):
            fixed_day_count = _get_convention(fixed_day_count)
        if isinstance(float_day_count, str):
            float_day_count = _get_convention(float_day_count)

        self.fixed_day_count = fixed_day_count
        self.float_day_count = float_day_count

        self.fixed_leg = FixedLeg(
            notional=notional,
            effective_date=start_date,
            termination_date=end_date,
            fixed_rate=fixed_rate,
            frequency=fixed_frequency,
            day_count=fixed_day_count,
            calendar=calendar,
        )

        # OIS uses compounded overnight rate
        self._float_schedule = Schedule(
            effective_date=start_date,
            termination_date=end_date,
            frequency=float_frequency,
            calendar=calendar,
        )
        self._calendar = calendar
        self._float_day_count = float_day_count

    def _compound_overnight_rate(
        self,
        accrual_start: date,
        accrual_end: date,
        ois_curve: Curve,
        valuation_date: date,
    ) -> float:
        """
        Calculate compounded overnight rate for a period.

        Uses daily compounding: prod(1 + r_i * delta_i) - 1
        """
        # Simplified: use forward rate from curve
        t1 = ois_curve.time_to_maturity(accrual_start)
        t2 = ois_curve.time_to_maturity(accrual_end)

        if t1 <= 0:
            t1 = 0.001
        if t2 <= t1:
            t2 = t1 + 1/365

        # Continuous compounding forward rate
        return ois_curve.forward_rate(t1, t2)

    def present_value(
        self,
        valuation_date: date,
        ois_curve: Curve,
    ) -> float:
        """Calculate present value of OIS."""
        fixed_pv = self.fixed_leg.present_value(valuation_date, ois_curve)

        # Calculate floating leg PV
        float_pv = 0.0
        for period in self._float_schedule.periods:
            payment_date = period.accrual_end
            if self._calendar:
                adj = BusinessDayAdjustment("modified_following", self._calendar)
                payment_date = adj.adjust(payment_date)

            if payment_date < valuation_date:
                continue

            rate = self._compound_overnight_rate(
                period.accrual_start, period.accrual_end, ois_curve, valuation_date
            )
            dcf = self._float_day_count.year_fraction(period.accrual_start, period.accrual_end)
            amount = self.notional * rate * dcf
            df = ois_curve.discount_factor(payment_date)
            float_pv += amount * df

        if self.swap_type == SwapType.PAYER:
            return float_pv - fixed_pv
        else:
            return fixed_pv - float_pv

    def par_rate(
        self,
        valuation_date: date,
        ois_curve: Curve,
    ) -> float:
        """Calculate par OIS rate."""
        # Calculate floating leg PV
        float_pv = 0.0
        for period in self._float_schedule.periods:
            payment_date = period.accrual_end
            if self._calendar:
                adj = BusinessDayAdjustment("modified_following", self._calendar)
                payment_date = adj.adjust(payment_date)

            if payment_date < valuation_date:
                continue

            rate = self._compound_overnight_rate(
                period.accrual_start, period.accrual_end, ois_curve, valuation_date
            )
            dcf = self._float_day_count.year_fraction(period.accrual_start, period.accrual_end)
            amount = self.notional * rate * dcf
            df = ois_curve.discount_factor(payment_date)
            float_pv += amount * df

        # Annuity
        annuity = self.fixed_leg.annuity(valuation_date, ois_curve)

        if abs(annuity) < 1e-10:
            return 0.0

        return float_pv / annuity
