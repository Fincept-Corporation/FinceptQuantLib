"""
Advanced Interest Rate Swaps

Provides sophisticated swap structures beyond vanilla IRS:
- Amortizing/Accreting Swaps: Variable notional schedules
- Compounded RFR Swaps: ISDA compounding methods (lookback, observation shift, lockout)
- Forward-Starting Swaps: Deferred effective date
- In-Arrears Swaps: Rate fixing at period end with convexity adjustment
- Capped/Floored Swaps: Embedded cap/floor options
- Range Accrual Swaps: Accrual based on rate range
- CMS Swaps: Constant Maturity Swap rates
- Callable/Puttable Swaps: Bermudan exercise features
- MTM Cross-Currency Swaps: Mark-to-market resets

Key Features:
- Full ISDA compounding method support for RFR
- Automatic convexity adjustments
- Multi-curve framework
- Automatic differentiation compatible
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Union, Optional, List, Tuple, Dict
import math

from finceptquantlib.core.foundations.types import (
    ExerciseType, NotionalSchedule,
)
from finceptquantlib.curves.curve import Curve
from finceptquantlib.scheduling.schedule import Schedule, Frequency, StubType
from finceptquantlib.scheduling.calendar import Calendar
from finceptquantlib.scheduling.daycount import DayCountConvention, _get_convention
from finceptquantlib.scheduling.adjustment import BusinessDayAdjustment


# =============================================================================
# Enums and Data Classes
# =============================================================================

class RFRCompoundingMethod(Enum):
    """ISDA RFR compounding methods."""
    FLAT = "flat"                       # No compounding
    COMPOUNDED = "compounded"           # Daily compounding
    AVERAGED = "averaged"               # Simple average
    WEIGHTED = "weighted"               # Weighted average by day count


class ObservationShift(Enum):
    """Observation shift method for RFR."""
    NONE = "none"                       # No shift (standard)
    OBSERVATION = "observation"         # Shift observation dates
    PAYMENT = "payment"                 # Shift payment calculation


@dataclass
class RFRFixing:
    """Single RFR fixing observation."""
    observation_date: date
    value_date: date
    rate: float
    weight: float = 1.0


@dataclass
class CompoundedPeriod:
    """A single compounded RFR period."""
    accrual_start: date
    accrual_end: date
    payment_date: date
    notional: float
    spread: float
    fixings: List[RFRFixing] = field(default_factory=list)
    compounded_rate: Optional[float] = None
    day_count_fraction: Optional[float] = None


@dataclass
class ExerciseSchedule:
    """Schedule for callable/puttable options."""
    exercise_dates: List[date]
    exercise_prices: Optional[List[float]] = None  # Strike prices/levels
    exercise_type: ExerciseType = ExerciseType.BERMUDAN

    def get_next_exercise(self, as_of: date) -> Optional[date]:
        """Get next available exercise date."""
        for ex_date in self.exercise_dates:
            if ex_date > as_of:
                return ex_date
        return None

    def get_exercise_price(self, ex_date: date) -> float:
        """Get exercise price for a date (default 0 = at par)."""
        if self.exercise_prices is None:
            return 0.0

        for i, d in enumerate(self.exercise_dates):
            if d == ex_date:
                return self.exercise_prices[i] if i < len(self.exercise_prices) else 0.0
        return 0.0


# =============================================================================
# Compounded RFR Swap
# =============================================================================

class CompoundedRFRSwap:
    """
    Swap with compounded Risk-Free Rate (RFR) floating leg.

    Implements ISDA standard compounding methods:
    - Lookback: Shifts observation window backward
    - Observation Shift: Adjusts weight calculation
    - Lockout: Uses rate from N days before payment
    - Payment Delay: Delay between accrual end and payment

    Examples:
        SOFR: 2 business day lookback
        SONIA: 5 business day lookback (shifted)
        ESTR: 2 business day payment delay
    """

    def __init__(
        self,
        notional: Union[float, NotionalSchedule],
        start_date: date,
        end_date: date,
        fixed_rate: float,
        pay_fixed: bool = True,
        spread: float = 0.0,
        # Compounding settings
        compounding_method: RFRCompoundingMethod = RFRCompoundingMethod.COMPOUNDED,
        observation_shift: ObservationShift = ObservationShift.NONE,
        lookback_days: int = 0,
        lockout_days: int = 0,
        payment_delay: int = 0,
        # Schedule settings
        fixed_frequency: Union[str, Frequency] = Frequency.ANNUAL,
        float_frequency: Union[str, Frequency] = Frequency.ANNUAL,
        fixed_day_count: Union[str, DayCountConvention] = "ACT/360",
        float_day_count: Union[str, DayCountConvention] = "ACT/360",
        calendar: Optional[Calendar] = None,
        stub_type: StubType = StubType.SHORT_FRONT,
        # Rate index
        index_name: str = "SOFR",
    ):
        """
        Initialize compounded RFR swap.

        Args:
            notional: Notional or schedule
            start_date: Effective date
            end_date: Maturity date
            fixed_rate: Fixed leg rate
            pay_fixed: True = payer, False = receiver
            spread: Spread on floating leg
            compounding_method: How to compound daily rates
            observation_shift: Observation date adjustment method
            lookback_days: Days to shift observation window
            lockout_days: Days before payment to lock rate
            payment_delay: Days between accrual end and payment
            fixed_frequency: Fixed leg payment frequency
            float_frequency: Float leg payment frequency
            fixed_day_count: Fixed leg day count
            float_day_count: Float leg day count
            calendar: Business day calendar
            stub_type: Stub period type
            index_name: RFR index name
        """
        self.start_date = start_date
        self.end_date = end_date
        self.fixed_rate = fixed_rate
        self.pay_fixed = pay_fixed
        self.spread = spread

        # Notional handling
        if isinstance(notional, NotionalSchedule):
            self.notional_schedule = notional
        else:
            self.notional_schedule = NotionalSchedule.constant(notional)

        # Compounding settings
        self.compounding_method = compounding_method
        self.observation_shift = observation_shift
        self.lookback_days = lookback_days
        self.lockout_days = lockout_days
        self.payment_delay = payment_delay

        # Day counts
        if isinstance(fixed_day_count, str):
            fixed_day_count = _get_convention(fixed_day_count)
        if isinstance(float_day_count, str):
            float_day_count = _get_convention(float_day_count)

        self.fixed_day_count = fixed_day_count
        self.float_day_count = float_day_count

        # Parse frequencies
        if isinstance(fixed_frequency, str):
            fixed_frequency = Frequency[fixed_frequency.upper()]
        if isinstance(float_frequency, str):
            float_frequency = Frequency[float_frequency.upper()]

        self.fixed_frequency = fixed_frequency
        self.float_frequency = float_frequency

        self.calendar = calendar
        self.stub_type = stub_type
        self.index_name = index_name

        # Generate schedules
        self._fixed_schedule = Schedule(
            effective_date=start_date,
            termination_date=end_date,
            frequency=fixed_frequency,
            calendar=calendar,
            stub=stub_type,
        )

        self._float_schedule = Schedule(
            effective_date=start_date,
            termination_date=end_date,
            frequency=float_frequency,
            calendar=calendar,
            stub=stub_type,
        )

    @property
    def initial_notional(self) -> float:
        """Get initial notional amount."""
        return self.notional_schedule.initial_notional

    def _get_observation_dates(
        self,
        accrual_start: date,
        accrual_end: date,
    ) -> Tuple[date, date]:
        """
        Calculate observation period dates with lookback/shift.

        Returns:
            (observation_start, observation_end) tuple
        """
        if self.observation_shift == ObservationShift.OBSERVATION:
            # Shift both observation dates
            obs_start = accrual_start - timedelta(days=self.lookback_days)
            obs_end = accrual_end - timedelta(days=self.lookback_days)
        else:
            # Standard lookback affects calculation, not observation period
            obs_start = accrual_start
            obs_end = accrual_end

        return obs_start, obs_end

    def _generate_daily_observations(
        self,
        accrual_start: date,
        accrual_end: date,
    ) -> List[Tuple[date, date, float]]:
        """
        Generate daily observation schedule.

        Returns:
            List of (observation_date, value_date, weight) tuples
        """
        obs_start, obs_end = self._get_observation_dates(accrual_start, accrual_end)

        observations = []
        current = obs_start

        # Calculate lockout boundary
        if self.lockout_days > 0:
            lockout_start = accrual_end - timedelta(days=self.lockout_days)
        else:
            lockout_start = accrual_end

        while current < obs_end:
            next_date = current + timedelta(days=1)

            # Skip weekends if calendar provided
            if self.calendar:
                while not self.calendar.is_business_day(next_date) and next_date < obs_end:
                    next_date += timedelta(days=1)

            # Calculate weight (day count fraction)
            weight = self.float_day_count.year_fraction(current, next_date)

            # Value date with lookback
            if self.observation_shift != ObservationShift.OBSERVATION:
                value_date = current - timedelta(days=self.lookback_days)
            else:
                value_date = current

            # Apply lockout (use last rate before lockout)
            if current >= lockout_start:
                # In lockout period - use lockout_start rate
                value_date = lockout_start - timedelta(days=self.lookback_days + 1)

            observations.append((current, value_date, weight))
            current = next_date

        return observations

    def _compound_rate(
        self,
        daily_rates: List[Tuple[float, float]],  # (rate, weight) pairs
    ) -> float:
        """
        Calculate compounded rate from daily observations.

        ISDA Compounding Formula:
        Compounded Rate = [∏(1 + r_i × d_i) - 1] / D
        where:
            r_i = overnight rate for day i
            d_i = day count fraction for day i (typically 1/360 or 1/365)
            D = total day count fraction for the period

        Args:
            daily_rates: List of (rate, day_count_fraction) pairs
        """
        if not daily_rates:
            return 0.0

        if self.compounding_method == RFRCompoundingMethod.FLAT:
            # Simple weighted average (no compounding effect)
            total_rate = sum(r * w for r, w in daily_rates)
            total_weight = sum(w for _, w in daily_rates)
            return total_rate / total_weight if total_weight > 0 else 0.0

        elif self.compounding_method == RFRCompoundingMethod.AVERAGED:
            # Simple arithmetic average (unweighted)
            return sum(r for r, _ in daily_rates) / len(daily_rates)

        elif self.compounding_method == RFRCompoundingMethod.WEIGHTED:
            # Day count weighted average
            total_rate = sum(r * w for r, w in daily_rates)
            total_weight = sum(w for _, w in daily_rates)
            return total_rate / total_weight if total_weight > 0 else 0.0

        else:  # COMPOUNDED (default ISDA method)
            # ISDA compounding: [∏(1 + r_i × d_i) - 1] / D
            # This gives the annualized compounded rate over the period
            product = 1.0
            total_dcf = 0.0

            for rate, dcf in daily_rates:
                product *= (1.0 + rate * dcf)
                total_dcf += dcf

            if total_dcf > 0:
                # Return the annualized compounded rate
                return (product - 1.0) / total_dcf
            return 0.0

    def _project_rfr_rate(
        self,
        observation_date: date,
        projection_curve: Curve,
    ) -> float:
        """Project overnight rate from curve."""
        t = projection_curve.time_to_maturity(observation_date)
        t_next = projection_curve.time_to_maturity(observation_date + timedelta(days=1))

        if t <= 0:
            t = 0.001
        if t_next <= t:
            t_next = t + 1/365

        return projection_curve.forward_rate(t, t_next)

    def _get_payment_date(self, accrual_end: date) -> date:
        """Calculate payment date with delay."""
        payment_date = accrual_end + timedelta(days=self.payment_delay)

        if self.calendar:
            adjustment = BusinessDayAdjustment("modified_following", self.calendar)
            payment_date = adjustment.adjust(payment_date)

        return payment_date

    def fixed_leg_pv(
        self,
        valuation_date: date,
        discount_curve: Curve,
    ) -> float:
        """Calculate fixed leg present value."""
        pv = 0.0

        for accrual_start, accrual_end in self._fixed_schedule.periods:
            payment_date = self._get_payment_date(accrual_end)

            if payment_date < valuation_date:
                continue

            notional = self.notional_schedule.get_notional(accrual_start)
            dcf = self.fixed_day_count.year_fraction(accrual_start, accrual_end)
            df = discount_curve.discount_factor(payment_date)

            pv += notional * self.fixed_rate * dcf * df

        return pv

    def float_leg_pv(
        self,
        valuation_date: date,
        projection_curve: Curve,
        discount_curve: Optional[Curve] = None,
        fixed_rates: Optional[Dict[date, float]] = None,
    ) -> float:
        """
        Calculate floating (compounded RFR) leg present value.

        Args:
            valuation_date: Valuation date
            projection_curve: Curve for projecting forward rates
            discount_curve: Discount curve (if different from projection)
            fixed_rates: Dictionary of historical fixings {date: rate}
        """
        if discount_curve is None:
            discount_curve = projection_curve

        fixed_rates = fixed_rates or {}
        pv = 0.0

        for accrual_start, accrual_end in self._float_schedule.periods:
            payment_date = self._get_payment_date(accrual_end)

            if payment_date < valuation_date:
                continue

            # Get daily observations
            observations = self._generate_daily_observations(accrual_start, accrual_end)

            # Get rate for each observation
            daily_rates = []
            for obs_date, value_date, weight in observations:
                if value_date in fixed_rates:
                    rate = fixed_rates[value_date]
                elif value_date <= valuation_date:
                    # Past fixing - should be provided, use projection as fallback
                    rate = self._project_rfr_rate(value_date, projection_curve)
                else:
                    # Future - project from curve
                    rate = self._project_rfr_rate(value_date, projection_curve)

                daily_rates.append((rate, weight))

            # Compound the daily rates
            compounded_rate = self._compound_rate(daily_rates)

            # Add spread
            total_rate = compounded_rate + self.spread

            # Calculate payment
            notional = self.notional_schedule.get_notional(accrual_start)
            dcf = self.float_day_count.year_fraction(accrual_start, accrual_end)
            df = discount_curve.discount_factor(payment_date)

            pv += notional * total_rate * dcf * df

        return pv

    def present_value(
        self,
        valuation_date: date,
        projection_curve: Curve,
        discount_curve: Optional[Curve] = None,
        fixed_rates: Optional[Dict[date, float]] = None,
    ) -> float:
        """Calculate swap NPV."""
        if discount_curve is None:
            discount_curve = projection_curve

        fixed_pv = self.fixed_leg_pv(valuation_date, discount_curve)
        float_pv = self.float_leg_pv(
            valuation_date, projection_curve, discount_curve, fixed_rates
        )

        if self.pay_fixed:
            return float_pv - fixed_pv
        else:
            return fixed_pv - float_pv

    def par_rate(
        self,
        valuation_date: date,
        projection_curve: Curve,
        discount_curve: Optional[Curve] = None,
    ) -> float:
        """Calculate par swap rate."""
        if discount_curve is None:
            discount_curve = projection_curve

        # Calculate float leg PV
        float_pv = self.float_leg_pv(valuation_date, projection_curve, discount_curve)

        # Calculate annuity
        annuity = 0.0
        for accrual_start, accrual_end in self._fixed_schedule.periods:
            payment_date = self._get_payment_date(accrual_end)

            if payment_date < valuation_date:
                continue

            notional = self.notional_schedule.get_notional(accrual_start)
            dcf = self.fixed_day_count.year_fraction(accrual_start, accrual_end)
            df = discount_curve.discount_factor(payment_date)

            annuity += notional * dcf * df

        if abs(annuity) < 1e-10:
            return 0.0

        return float_pv / annuity

    def dv01(
        self,
        valuation_date: date,
        projection_curve: Curve,
        discount_curve: Optional[Curve] = None,
        bump_size: float = 1.0,
    ) -> float:
        """Calculate DV01 (sensitivity to 1bp parallel shift)."""
        if discount_curve is None:
            discount_curve = projection_curve

        base_pv = self.present_value(valuation_date, projection_curve, discount_curve)

        # Bump curves
        proj_up = projection_curve.shift(bump_size)
        disc_up = discount_curve.shift(bump_size) if discount_curve != projection_curve else proj_up

        bumped_pv = self.present_value(valuation_date, proj_up, disc_up)

        return bumped_pv - base_pv

    def __repr__(self) -> str:
        swap_type = "Payer" if self.pay_fixed else "Receiver"
        return (
            f"CompoundedRFRSwap({swap_type}, "
            f"notional={self.initial_notional:,.0f}, "
            f"rate={self.fixed_rate:.4%}, "
            f"index={self.index_name}, "
            f"method={self.compounding_method.value}, "
            f"{self.start_date} to {self.end_date})"
        )


# =============================================================================
# Amortizing Swap
# =============================================================================

class AmortizingSwap:
    """
    Interest Rate Swap with variable notional schedule.

    Supports:
    - Linear amortization
    - Mortgage-style (constant payment)
    - Custom amortization schedules
    - Accreting (increasing) notional
    """

    def __init__(
        self,
        notional_schedule: NotionalSchedule,
        start_date: date,
        end_date: date,
        fixed_rate: float,
        pay_fixed: bool = True,
        index_tenor: str = "3M",
        spread: float = 0.0,
        fixed_frequency: Union[str, Frequency] = Frequency.SEMI_ANNUAL,
        float_frequency: Union[str, Frequency] = Frequency.QUARTERLY,
        fixed_day_count: Union[str, DayCountConvention] = "30/360",
        float_day_count: Union[str, DayCountConvention] = "ACT/360",
        calendar: Optional[Calendar] = None,
        include_principal_exchange: bool = False,
    ):
        """
        Initialize amortizing swap.

        Args:
            notional_schedule: Notional schedule (use NotionalSchedule.* methods)
            start_date: Effective date
            end_date: Maturity date
            fixed_rate: Fixed leg rate
            pay_fixed: True = payer, False = receiver
            index_tenor: Floating rate index tenor
            spread: Spread over floating rate
            fixed_frequency: Fixed leg payment frequency
            float_frequency: Float leg payment frequency
            fixed_day_count: Fixed leg day count
            float_day_count: Float leg day count
            calendar: Business day calendar
            include_principal_exchange: If True, exchange notional changes
        """
        self.notional_schedule = notional_schedule
        self.start_date = start_date
        self.end_date = end_date
        self.fixed_rate = fixed_rate
        self.pay_fixed = pay_fixed
        self.index_tenor = index_tenor
        self.spread = spread
        self.include_principal_exchange = include_principal_exchange

        # Day counts
        if isinstance(fixed_day_count, str):
            fixed_day_count = _get_convention(fixed_day_count)
        if isinstance(float_day_count, str):
            float_day_count = _get_convention(float_day_count)

        self.fixed_day_count = fixed_day_count
        self.float_day_count = float_day_count

        # Parse frequencies
        if isinstance(fixed_frequency, str):
            fixed_frequency = Frequency[fixed_frequency.upper()]
        if isinstance(float_frequency, str):
            float_frequency = Frequency[float_frequency.upper()]

        self.fixed_frequency = fixed_frequency
        self.float_frequency = float_frequency

        self.calendar = calendar

        # Parse index tenor
        self._index_tenor_years = self._parse_tenor(index_tenor)

        # Generate schedules
        self._fixed_schedule = Schedule(
            effective_date=start_date,
            termination_date=end_date,
            frequency=fixed_frequency,
            calendar=calendar,
        )

        self._float_schedule = Schedule(
            effective_date=start_date,
            termination_date=end_date,
            frequency=float_frequency,
            calendar=calendar,
        )

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

    @classmethod
    def linear(
        cls,
        initial_notional: float,
        final_notional: float,
        start_date: date,
        end_date: date,
        fixed_rate: float,
        amort_frequency: Frequency = Frequency.ANNUAL,
        **kwargs,
    ) -> "AmortizingSwap":
        """Create linearly amortizing swap."""
        schedule = NotionalSchedule.linear_amortizing(
            initial=initial_notional,
            final=final_notional,
            start_date=start_date,
            end_date=end_date,
            frequency=amort_frequency,
        )
        return cls(
            notional_schedule=schedule,
            start_date=start_date,
            end_date=end_date,
            fixed_rate=fixed_rate,
            **kwargs,
        )

    @classmethod
    def mortgage(
        cls,
        initial_notional: float,
        rate: float,
        start_date: date,
        end_date: date,
        fixed_rate: float,
        amort_frequency: Frequency = Frequency.MONTHLY,
        **kwargs,
    ) -> "AmortizingSwap":
        """Create mortgage-style amortizing swap."""
        schedule = NotionalSchedule.mortgage_style(
            initial=initial_notional,
            rate=rate,
            start_date=start_date,
            end_date=end_date,
            frequency=amort_frequency,
        )
        return cls(
            notional_schedule=schedule,
            start_date=start_date,
            end_date=end_date,
            fixed_rate=fixed_rate,
            **kwargs,
        )

    @property
    def initial_notional(self) -> float:
        """Get initial notional."""
        return self.notional_schedule.initial_notional

    def _get_payment_date(self, accrual_end: date) -> date:
        """Get payment date from accrual end."""
        if self.calendar:
            adjustment = BusinessDayAdjustment("modified_following", self.calendar)
            return adjustment.adjust(accrual_end)
        return accrual_end

    def _project_rate(
        self,
        accrual_start: date,
        accrual_end: date,
        projection_curve: Curve,
    ) -> float:
        """Project forward rate from curve."""
        t1 = projection_curve.time_to_maturity(accrual_start)
        t2 = projection_curve.time_to_maturity(accrual_end)

        if t1 <= 0:
            t1 = 0.001
        if t2 <= t1:
            t2 = t1 + self._index_tenor_years

        return projection_curve.forward_rate(t1, t2)

    def fixed_leg_pv(
        self,
        valuation_date: date,
        discount_curve: Curve,
    ) -> float:
        """Calculate fixed leg present value."""
        pv = 0.0
        prev_notional = self.initial_notional

        for accrual_start, accrual_end in self._fixed_schedule.periods:
            payment_date = self._get_payment_date(accrual_end)

            if payment_date < valuation_date:
                prev_notional = self.notional_schedule.get_notional(accrual_end)
                continue

            notional = self.notional_schedule.get_notional(accrual_start)
            dcf = self.fixed_day_count.year_fraction(accrual_start, accrual_end)
            df = discount_curve.discount_factor(payment_date)

            # Interest payment
            pv += notional * self.fixed_rate * dcf * df

            # Principal exchange (amortization)
            if self.include_principal_exchange:
                new_notional = self.notional_schedule.get_notional(accrual_end)
                principal_payment = prev_notional - new_notional
                pv += principal_payment * df
                prev_notional = new_notional

        return pv

    def float_leg_pv(
        self,
        valuation_date: date,
        projection_curve: Curve,
        discount_curve: Optional[Curve] = None,
    ) -> float:
        """Calculate floating leg present value."""
        if discount_curve is None:
            discount_curve = projection_curve

        pv = 0.0
        prev_notional = self.initial_notional

        for accrual_start, accrual_end in self._float_schedule.periods:
            payment_date = self._get_payment_date(accrual_end)

            if payment_date < valuation_date:
                prev_notional = self.notional_schedule.get_notional(accrual_end)
                continue

            notional = self.notional_schedule.get_notional(accrual_start)
            rate = self._project_rate(accrual_start, accrual_end, projection_curve)
            dcf = self.float_day_count.year_fraction(accrual_start, accrual_end)
            df = discount_curve.discount_factor(payment_date)

            # Interest payment
            pv += notional * (rate + self.spread) * dcf * df

            # Principal exchange
            if self.include_principal_exchange:
                new_notional = self.notional_schedule.get_notional(accrual_end)
                principal_payment = prev_notional - new_notional
                pv += principal_payment * df
                prev_notional = new_notional

        return pv

    def present_value(
        self,
        valuation_date: date,
        projection_curve: Curve,
        discount_curve: Optional[Curve] = None,
    ) -> float:
        """Calculate swap NPV."""
        if discount_curve is None:
            discount_curve = projection_curve

        fixed_pv = self.fixed_leg_pv(valuation_date, discount_curve)
        float_pv = self.float_leg_pv(valuation_date, projection_curve, discount_curve)

        if self.pay_fixed:
            return float_pv - fixed_pv
        else:
            return fixed_pv - float_pv

    def weighted_average_life(self, valuation_date: date) -> float:
        """
        Calculate weighted average life (WAL).

        WAL = sum(principal_i * time_i) / total_principal
        """
        total_weighted = 0.0
        total_principal = 0.0

        prev_notional = self.initial_notional

        for _, accrual_end in self._fixed_schedule.periods:
            if accrual_end < valuation_date:
                prev_notional = self.notional_schedule.get_notional(accrual_end)
                continue

            new_notional = self.notional_schedule.get_notional(accrual_end)
            principal = prev_notional - new_notional

            if principal > 0:
                years = (accrual_end - valuation_date).days / 365.0
                total_weighted += principal * years
                total_principal += principal

            prev_notional = new_notional

        # Add final principal
        if prev_notional > 0:
            years = (self.end_date - valuation_date).days / 365.0
            total_weighted += prev_notional * years
            total_principal += prev_notional

        if total_principal > 0:
            return total_weighted / total_principal
        return 0.0

    def __repr__(self) -> str:
        swap_type = "Payer" if self.pay_fixed else "Receiver"
        return (
            f"AmortizingSwap({swap_type}, "
            f"initial={self.initial_notional:,.0f}, "
            f"rate={self.fixed_rate:.4%}, "
            f"{self.start_date} to {self.end_date})"
        )


# =============================================================================
# Capped/Floored Swap
# =============================================================================

class CappedFlooredSwap:
    """
    Interest Rate Swap with embedded cap and/or floor on floating leg.

    The floating rate is bounded:
    - Cap: max(floating_rate, cap_rate) = cap_rate when rate > cap
    - Floor: min(floating_rate, floor_rate) = floor_rate when rate < floor
    - Collar: Both cap and floor

    Pricing requires volatility surface for option valuation.
    """

    def __init__(
        self,
        notional: float,
        start_date: date,
        end_date: date,
        fixed_rate: float,
        pay_fixed: bool = True,
        cap_rate: Optional[float] = None,
        floor_rate: Optional[float] = None,
        index_tenor: str = "3M",
        spread: float = 0.0,
        fixed_frequency: Union[str, Frequency] = Frequency.SEMI_ANNUAL,
        float_frequency: Union[str, Frequency] = Frequency.QUARTERLY,
        fixed_day_count: Union[str, DayCountConvention] = "30/360",
        float_day_count: Union[str, DayCountConvention] = "ACT/360",
        calendar: Optional[Calendar] = None,
    ):
        """
        Initialize capped/floored swap.

        Args:
            notional: Notional principal
            start_date: Effective date
            end_date: Maturity date
            fixed_rate: Fixed leg rate
            pay_fixed: True = payer (pay fixed, receive capped/floored floating)
            cap_rate: Cap strike (None = no cap)
            floor_rate: Floor strike (None = no floor)
            index_tenor: Floating rate index tenor
            spread: Spread over floating rate
            fixed_frequency: Fixed leg payment frequency
            float_frequency: Float leg payment frequency
            fixed_day_count: Fixed leg day count
            float_day_count: Float leg day count
            calendar: Business day calendar
        """
        self.notional = notional
        self.start_date = start_date
        self.end_date = end_date
        self.fixed_rate = fixed_rate
        self.pay_fixed = pay_fixed
        self.cap_rate = cap_rate
        self.floor_rate = floor_rate
        self.index_tenor = index_tenor
        self.spread = spread

        # Day counts
        if isinstance(fixed_day_count, str):
            fixed_day_count = _get_convention(fixed_day_count)
        if isinstance(float_day_count, str):
            float_day_count = _get_convention(float_day_count)

        self.fixed_day_count = fixed_day_count
        self.float_day_count = float_day_count

        # Parse frequencies
        if isinstance(fixed_frequency, str):
            fixed_frequency = Frequency[fixed_frequency.upper()]
        if isinstance(float_frequency, str):
            float_frequency = Frequency[float_frequency.upper()]

        self.fixed_frequency = fixed_frequency
        self.float_frequency = float_frequency

        self.calendar = calendar

        # Parse index tenor
        self._index_tenor_years = self._parse_tenor(index_tenor)

        # Generate schedules
        self._fixed_schedule = Schedule(
            effective_date=start_date,
            termination_date=end_date,
            frequency=fixed_frequency,
            calendar=calendar,
        )

        self._float_schedule = Schedule(
            effective_date=start_date,
            termination_date=end_date,
            frequency=float_frequency,
            calendar=calendar,
        )

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
        return 0.25  # Default 3M

    @property
    def is_capped(self) -> bool:
        """Check if swap has cap."""
        return self.cap_rate is not None

    @property
    def is_floored(self) -> bool:
        """Check if swap has floor."""
        return self.floor_rate is not None

    @property
    def is_collared(self) -> bool:
        """Check if swap has both cap and floor (collar)."""
        return self.is_capped and self.is_floored

    def _get_payment_date(self, accrual_end: date) -> date:
        """Get payment date from accrual end."""
        if self.calendar:
            adjustment = BusinessDayAdjustment("modified_following", self.calendar)
            return adjustment.adjust(accrual_end)
        return accrual_end

    def _project_rate(
        self,
        accrual_start: date,
        accrual_end: date,
        projection_curve: Curve,
    ) -> float:
        """Project forward rate from curve."""
        t1 = projection_curve.time_to_maturity(accrual_start)
        t2 = projection_curve.time_to_maturity(accrual_end)

        if t1 <= 0:
            t1 = 0.001
        if t2 <= t1:
            t2 = t1 + self._index_tenor_years

        return projection_curve.forward_rate(t1, t2)

    def _apply_cap_floor(self, rate: float) -> float:
        """Apply cap and/or floor to rate."""
        if self.cap_rate is not None:
            rate = min(rate, self.cap_rate)
        if self.floor_rate is not None:
            rate = max(rate, self.floor_rate)
        return rate

    def _black_caplet_price(
        self,
        forward: float,
        strike: float,
        volatility: float,
        time_to_expiry: float,
        dcf: float,
        discount_factor: float,
        is_cap: bool = True,
    ) -> float:
        """
        Price a caplet/floorlet using Black's formula (1976).

        Black's formula for interest rate options:
        Caplet = N × τ × DF × [F × N(d1) - K × N(d2)]
        Floorlet = N × τ × DF × [K × N(-d2) - F × N(-d1)]

        where:
            N = notional
            τ = day count fraction (accrual period)
            DF = discount factor to payment date
            F = forward rate
            K = strike rate
            d1 = [ln(F/K) + 0.5 × σ² × T] / (σ × √T)
            d2 = d1 - σ × √T
            T = time to option expiry
            σ = volatility

        Args:
            forward: Forward rate
            strike: Strike rate
            volatility: Implied volatility (lognormal)
            time_to_expiry: Time to option expiry in years
            dcf: Day count fraction for payment period
            discount_factor: Discount factor to payment date
            is_cap: True for caplet, False for floorlet
        """
        # Handle edge cases
        if forward <= 0 or strike <= 0:
            return 0.0

        if time_to_expiry <= 0 or volatility <= 0:
            # Intrinsic value only (expired or zero vol)
            if is_cap:
                return self.notional * dcf * discount_factor * max(0, forward - strike)
            else:
                return self.notional * dcf * discount_factor * max(0, strike - forward)

        # Black's formula
        sqrt_t = math.sqrt(time_to_expiry)
        vol_sqrt_t = volatility * sqrt_t

        if vol_sqrt_t < 1e-10:
            # Near-zero vol - use intrinsic
            if is_cap:
                return self.notional * dcf * discount_factor * max(0, forward - strike)
            else:
                return self.notional * dcf * discount_factor * max(0, strike - forward)

        d1 = (math.log(forward / strike) + 0.5 * volatility ** 2 * time_to_expiry) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t

        # Standard normal CDF using Abramowitz & Stegun approximation (accurate to ~1e-7)
        def norm_cdf(x: float) -> float:
            """Approximation of standard normal CDF."""
            a1 = 0.254829592
            a2 = -0.284496736
            a3 = 1.421413741
            a4 = -1.453152027
            a5 = 1.061405429
            p = 0.3275911

            sign = 1 if x >= 0 else -1
            x = abs(x) / math.sqrt(2)

            t = 1.0 / (1.0 + p * x)
            y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

            return 0.5 * (1.0 + sign * y)

        # Calculate option value
        if is_cap:
            # Caplet: pays max(F - K, 0)
            option_value = forward * norm_cdf(d1) - strike * norm_cdf(d2)
        else:
            # Floorlet: pays max(K - F, 0)
            option_value = strike * norm_cdf(-d2) - forward * norm_cdf(-d1)

        return self.notional * dcf * discount_factor * option_value

    def fixed_leg_pv(
        self,
        valuation_date: date,
        discount_curve: Curve,
    ) -> float:
        """Calculate fixed leg present value."""
        pv = 0.0

        for accrual_start, accrual_end in self._fixed_schedule.periods:
            payment_date = self._get_payment_date(accrual_end)

            if payment_date < valuation_date:
                continue

            dcf = self.fixed_day_count.year_fraction(accrual_start, accrual_end)
            df = discount_curve.discount_factor(payment_date)

            pv += self.notional * self.fixed_rate * dcf * df

        return pv

    def float_leg_pv(
        self,
        valuation_date: date,
        projection_curve: Curve,
        discount_curve: Optional[Curve] = None,
        volatility: float = 0.20,
    ) -> float:
        """
        Calculate capped/floored floating leg present value.

        Args:
            valuation_date: Valuation date
            projection_curve: Curve for projecting forward rates
            discount_curve: Discount curve
            volatility: Flat volatility for cap/floor pricing
        """
        if discount_curve is None:
            discount_curve = projection_curve

        pv = 0.0

        for accrual_start, accrual_end in self._float_schedule.periods:
            payment_date = self._get_payment_date(accrual_end)

            if payment_date < valuation_date:
                continue

            forward = self._project_rate(accrual_start, accrual_end, projection_curve)
            dcf = self.float_day_count.year_fraction(accrual_start, accrual_end)
            df = discount_curve.discount_factor(payment_date)

            # Time to fixing (approximation)
            time_to_expiry = (accrual_start - valuation_date).days / 365.0

            # Base floating payment (before cap/floor)
            base_rate = forward + self.spread

            # Apply deterministic cap/floor or price optionality
            if time_to_expiry <= 0:
                # Already fixed - apply cap/floor deterministically
                effective_rate = self._apply_cap_floor(base_rate)
                pv += self.notional * effective_rate * dcf * df
            else:
                # Future fixing - use forward with optional adjustment
                # For simplicity, use forward rate adjusted by cap/floor
                # Full implementation would price caplets/floorlets separately

                # Linear swap value
                pv += self.notional * base_rate * dcf * df

                # Cap value (short position for receiver)
                if self.cap_rate is not None:
                    cap_value = self._black_caplet_price(
                        forward=forward + self.spread,
                        strike=self.cap_rate,
                        volatility=volatility,
                        time_to_expiry=time_to_expiry,
                        dcf=dcf,
                        discount_factor=df,
                        is_cap=True,
                    )
                    pv -= cap_value  # Cap reduces value for receiver

                # Floor value (long position for receiver)
                if self.floor_rate is not None:
                    floor_value = self._black_caplet_price(
                        forward=forward + self.spread,
                        strike=self.floor_rate,
                        volatility=volatility,
                        time_to_expiry=time_to_expiry,
                        dcf=dcf,
                        discount_factor=df,
                        is_cap=False,
                    )
                    pv += floor_value  # Floor adds value for receiver

        return pv

    def present_value(
        self,
        valuation_date: date,
        projection_curve: Curve,
        discount_curve: Optional[Curve] = None,
        volatility: float = 0.20,
    ) -> float:
        """Calculate swap NPV."""
        if discount_curve is None:
            discount_curve = projection_curve

        fixed_pv = self.fixed_leg_pv(valuation_date, discount_curve)
        float_pv = self.float_leg_pv(
            valuation_date, projection_curve, discount_curve, volatility
        )

        if self.pay_fixed:
            return float_pv - fixed_pv
        else:
            return fixed_pv - float_pv

    def cap_value(
        self,
        valuation_date: date,
        projection_curve: Curve,
        discount_curve: Optional[Curve] = None,
        volatility: float = 0.20,
    ) -> float:
        """Calculate standalone cap value."""
        if not self.is_capped:
            return 0.0

        if discount_curve is None:
            discount_curve = projection_curve

        total = 0.0

        for accrual_start, accrual_end in self._float_schedule.periods:
            payment_date = self._get_payment_date(accrual_end)

            if payment_date < valuation_date:
                continue

            forward = self._project_rate(accrual_start, accrual_end, projection_curve)
            dcf = self.float_day_count.year_fraction(accrual_start, accrual_end)
            df = discount_curve.discount_factor(payment_date)
            time_to_expiry = (accrual_start - valuation_date).days / 365.0

            caplet = self._black_caplet_price(
                forward=forward + self.spread,
                strike=self.cap_rate,
                volatility=volatility,
                time_to_expiry=time_to_expiry,
                dcf=dcf,
                discount_factor=df,
                is_cap=True,
            )
            total += caplet

        return total

    def floor_value(
        self,
        valuation_date: date,
        projection_curve: Curve,
        discount_curve: Optional[Curve] = None,
        volatility: float = 0.20,
    ) -> float:
        """Calculate standalone floor value."""
        if not self.is_floored:
            return 0.0

        if discount_curve is None:
            discount_curve = projection_curve

        total = 0.0

        for accrual_start, accrual_end in self._float_schedule.periods:
            payment_date = self._get_payment_date(accrual_end)

            if payment_date < valuation_date:
                continue

            forward = self._project_rate(accrual_start, accrual_end, projection_curve)
            dcf = self.float_day_count.year_fraction(accrual_start, accrual_end)
            df = discount_curve.discount_factor(payment_date)
            time_to_expiry = (accrual_start - valuation_date).days / 365.0

            floorlet = self._black_caplet_price(
                forward=forward + self.spread,
                strike=self.floor_rate,
                volatility=volatility,
                time_to_expiry=time_to_expiry,
                dcf=dcf,
                discount_factor=df,
                is_cap=False,
            )
            total += floorlet

        return total

    def __repr__(self) -> str:
        swap_type = "Payer" if self.pay_fixed else "Receiver"
        cap_str = f", cap={self.cap_rate:.2%}" if self.is_capped else ""
        floor_str = f", floor={self.floor_rate:.2%}" if self.is_floored else ""
        return (
            f"CappedFlooredSwap({swap_type}, "
            f"notional={self.notional:,.0f}, "
            f"rate={self.fixed_rate:.4%}{cap_str}{floor_str}, "
            f"{self.start_date} to {self.end_date})"
        )


# =============================================================================
# Range Accrual Swap
# =============================================================================

class RangeAccrualSwap:
    """
    Swap where floating leg accrues only when rate is within a specified range.

    The floating leg pays:
        Notional * (rate + spread) * DCF * (days_in_range / total_days)

    This is a path-dependent exotic structure.
    """

    def __init__(
        self,
        notional: float,
        start_date: date,
        end_date: date,
        fixed_rate: float,
        pay_fixed: bool = True,
        lower_bound: float = 0.0,
        upper_bound: float = 0.10,
        index_tenor: str = "3M",
        spread: float = 0.0,
        observation_index: str = "3M",  # Index to observe for range
        fixed_frequency: Union[str, Frequency] = Frequency.SEMI_ANNUAL,
        float_frequency: Union[str, Frequency] = Frequency.QUARTERLY,
        fixed_day_count: Union[str, DayCountConvention] = "30/360",
        float_day_count: Union[str, DayCountConvention] = "ACT/360",
        calendar: Optional[Calendar] = None,
    ):
        """
        Initialize range accrual swap.

        Args:
            notional: Notional principal
            start_date: Effective date
            end_date: Maturity date
            fixed_rate: Fixed leg rate
            pay_fixed: True = payer
            lower_bound: Lower range boundary (inclusive)
            upper_bound: Upper range boundary (inclusive)
            index_tenor: Floating rate index tenor (for payment)
            spread: Spread over floating rate
            observation_index: Index to observe for range condition
            fixed_frequency: Fixed leg payment frequency
            float_frequency: Float leg payment frequency
            fixed_day_count: Fixed leg day count
            float_day_count: Float leg day count
            calendar: Business day calendar
        """
        self.notional = notional
        self.start_date = start_date
        self.end_date = end_date
        self.fixed_rate = fixed_rate
        self.pay_fixed = pay_fixed
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.index_tenor = index_tenor
        self.spread = spread
        self.observation_index = observation_index

        # Day counts
        if isinstance(fixed_day_count, str):
            fixed_day_count = _get_convention(fixed_day_count)
        if isinstance(float_day_count, str):
            float_day_count = _get_convention(float_day_count)

        self.fixed_day_count = fixed_day_count
        self.float_day_count = float_day_count

        # Parse frequencies
        if isinstance(fixed_frequency, str):
            fixed_frequency = Frequency[fixed_frequency.upper()]
        if isinstance(float_frequency, str):
            float_frequency = Frequency[float_frequency.upper()]

        self.fixed_frequency = fixed_frequency
        self.float_frequency = float_frequency

        self.calendar = calendar

        # Parse index tenor
        self._index_tenor_years = self._parse_tenor(index_tenor)
        self._obs_tenor_years = self._parse_tenor(observation_index)

        # Generate schedules
        self._fixed_schedule = Schedule(
            effective_date=start_date,
            termination_date=end_date,
            frequency=fixed_frequency,
            calendar=calendar,
        )

        self._float_schedule = Schedule(
            effective_date=start_date,
            termination_date=end_date,
            frequency=float_frequency,
            calendar=calendar,
        )

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
        return 0.25

    def _get_payment_date(self, accrual_end: date) -> date:
        """Get payment date from accrual end."""
        if self.calendar:
            adjustment = BusinessDayAdjustment("modified_following", self.calendar)
            return adjustment.adjust(accrual_end)
        return accrual_end

    def _project_rate(
        self,
        accrual_start: date,
        accrual_end: date,
        tenor_years: float,
        projection_curve: Curve,
    ) -> float:
        """Project forward rate from curve."""
        t1 = projection_curve.time_to_maturity(accrual_start)
        t2 = projection_curve.time_to_maturity(accrual_end)

        if t1 <= 0:
            t1 = 0.001
        if t2 <= t1:
            t2 = t1 + tenor_years

        return projection_curve.forward_rate(t1, t2)

    def _probability_in_range(
        self,
        forward: float,
        volatility: float,
        time_to_expiry: float,
    ) -> float:
        """
        Calculate probability that rate will be in range [lower, upper].

        Uses lognormal distribution assumption.
        """
        if time_to_expiry <= 0:
            # Already observed
            return 1.0 if self.lower_bound <= forward <= self.upper_bound else 0.0

        if volatility <= 0:
            return 1.0 if self.lower_bound <= forward <= self.upper_bound else 0.0

        # Lognormal CDF for rate < X: N(d2) where d2 = (ln(F/X) + 0.5*vol^2*T) / (vol*sqrt(T))
        def norm_cdf(x: float) -> float:
            """Approximation of standard normal CDF."""
            a1 = 0.254829592
            a2 = -0.284496736
            a3 = 1.421413741
            a4 = -1.453152027
            a5 = 1.061405429
            p = 0.3275911

            sign = 1 if x >= 0 else -1
            x = abs(x) / math.sqrt(2)

            t = 1.0 / (1.0 + p * x)
            y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

            return 0.5 * (1.0 + sign * y)

        sqrt_t = math.sqrt(time_to_expiry)
        vol_sqrt_t = volatility * sqrt_t

        # P(rate < upper)
        if self.upper_bound > 0:
            d2_upper = (math.log(forward / self.upper_bound) + 0.5 * volatility ** 2 * time_to_expiry) / vol_sqrt_t
            p_upper = norm_cdf(d2_upper)
        else:
            p_upper = 0.0

        # P(rate < lower)
        if self.lower_bound > 0:
            d2_lower = (math.log(forward / self.lower_bound) + 0.5 * volatility ** 2 * time_to_expiry) / vol_sqrt_t
            p_lower = norm_cdf(d2_lower)
        else:
            p_lower = 0.0

        # P(lower <= rate <= upper) = P(rate <= upper) - P(rate < lower)
        return max(0.0, min(1.0, p_upper - p_lower))

    def fixed_leg_pv(
        self,
        valuation_date: date,
        discount_curve: Curve,
    ) -> float:
        """Calculate fixed leg present value."""
        pv = 0.0

        for accrual_start, accrual_end in self._fixed_schedule.periods:
            payment_date = self._get_payment_date(accrual_end)

            if payment_date < valuation_date:
                continue

            dcf = self.fixed_day_count.year_fraction(accrual_start, accrual_end)
            df = discount_curve.discount_factor(payment_date)

            pv += self.notional * self.fixed_rate * dcf * df

        return pv

    def float_leg_pv(
        self,
        valuation_date: date,
        projection_curve: Curve,
        discount_curve: Optional[Curve] = None,
        volatility: float = 0.20,
    ) -> float:
        """
        Calculate range accrual floating leg present value.

        Args:
            valuation_date: Valuation date
            projection_curve: Curve for projecting forward rates
            discount_curve: Discount curve
            volatility: Volatility for range probability calculation
        """
        if discount_curve is None:
            discount_curve = projection_curve

        pv = 0.0

        for accrual_start, accrual_end in self._float_schedule.periods:
            payment_date = self._get_payment_date(accrual_end)

            if payment_date < valuation_date:
                continue

            # Payment rate
            payment_rate = self._project_rate(
                accrual_start, accrual_end, self._index_tenor_years, projection_curve
            )

            # Observation rate (for range check)
            obs_rate = self._project_rate(
                accrual_start, accrual_end, self._obs_tenor_years, projection_curve
            )

            dcf = self.float_day_count.year_fraction(accrual_start, accrual_end)
            df = discount_curve.discount_factor(payment_date)

            # Time to observation
            time_to_expiry = (accrual_start - valuation_date).days / 365.0

            # Calculate probability of being in range
            prob_in_range = self._probability_in_range(obs_rate, volatility, time_to_expiry)

            # Expected payment = rate * probability_in_range
            expected_rate = (payment_rate + self.spread) * prob_in_range

            pv += self.notional * expected_rate * dcf * df

        return pv

    def present_value(
        self,
        valuation_date: date,
        projection_curve: Curve,
        discount_curve: Optional[Curve] = None,
        volatility: float = 0.20,
    ) -> float:
        """Calculate swap NPV."""
        if discount_curve is None:
            discount_curve = projection_curve

        fixed_pv = self.fixed_leg_pv(valuation_date, discount_curve)
        float_pv = self.float_leg_pv(
            valuation_date, projection_curve, discount_curve, volatility
        )

        if self.pay_fixed:
            return float_pv - fixed_pv
        else:
            return fixed_pv - float_pv

    def __repr__(self) -> str:
        swap_type = "Payer" if self.pay_fixed else "Receiver"
        return (
            f"RangeAccrualSwap({swap_type}, "
            f"notional={self.notional:,.0f}, "
            f"rate={self.fixed_rate:.4%}, "
            f"range=[{self.lower_bound:.2%}, {self.upper_bound:.2%}], "
            f"{self.start_date} to {self.end_date})"
        )


# =============================================================================
# CMS Swap (Constant Maturity Swap)
# =============================================================================

class CMSSwap:
    """
    Constant Maturity Swap.

    Exchange of fixed rate for a CMS rate (swap rate of specified tenor).
    E.g., pay fixed, receive 10Y swap rate.

    Requires convexity adjustment due to the different nature of
    forward swap rates vs forward LIBOR rates.
    """

    def __init__(
        self,
        notional: float,
        start_date: date,
        end_date: date,
        fixed_rate: float,
        pay_fixed: bool = True,
        cms_tenor: str = "10Y",
        spread: float = 0.0,
        fixed_frequency: Union[str, Frequency] = Frequency.SEMI_ANNUAL,
        float_frequency: Union[str, Frequency] = Frequency.QUARTERLY,
        fixed_day_count: Union[str, DayCountConvention] = "30/360",
        float_day_count: Union[str, DayCountConvention] = "ACT/360",
        calendar: Optional[Calendar] = None,
    ):
        """
        Initialize CMS swap.

        Args:
            notional: Notional principal
            start_date: Effective date
            end_date: Maturity date
            fixed_rate: Fixed leg rate
            pay_fixed: True = payer
            cms_tenor: Underlying swap tenor (e.g., "5Y", "10Y", "30Y")
            spread: Spread over CMS rate
            fixed_frequency: Fixed leg payment frequency
            float_frequency: Float leg payment frequency (CMS reset frequency)
            fixed_day_count: Fixed leg day count
            float_day_count: Float leg day count
            calendar: Business day calendar
        """
        self.notional = notional
        self.start_date = start_date
        self.end_date = end_date
        self.fixed_rate = fixed_rate
        self.pay_fixed = pay_fixed
        self.cms_tenor = cms_tenor
        self.spread = spread

        # Day counts
        if isinstance(fixed_day_count, str):
            fixed_day_count = _get_convention(fixed_day_count)
        if isinstance(float_day_count, str):
            float_day_count = _get_convention(float_day_count)

        self.fixed_day_count = fixed_day_count
        self.float_day_count = float_day_count

        # Parse frequencies
        if isinstance(fixed_frequency, str):
            fixed_frequency = Frequency[fixed_frequency.upper()]
        if isinstance(float_frequency, str):
            float_frequency = Frequency[float_frequency.upper()]

        self.fixed_frequency = fixed_frequency
        self.float_frequency = float_frequency

        self.calendar = calendar

        # Parse CMS tenor
        self._cms_tenor_years = self._parse_tenor(cms_tenor)

        # Generate schedules
        self._fixed_schedule = Schedule(
            effective_date=start_date,
            termination_date=end_date,
            frequency=fixed_frequency,
            calendar=calendar,
        )

        self._float_schedule = Schedule(
            effective_date=start_date,
            termination_date=end_date,
            frequency=float_frequency,
            calendar=calendar,
        )

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
        return 10.0  # Default 10Y

    def _get_payment_date(self, accrual_end: date) -> date:
        """Get payment date from accrual end."""
        if self.calendar:
            adjustment = BusinessDayAdjustment("modified_following", self.calendar)
            return adjustment.adjust(accrual_end)
        return accrual_end

    def _forward_swap_rate(
        self,
        forward_start: date,
        projection_curve: Curve,
    ) -> float:
        """
        Calculate forward starting swap rate.

        Args:
            forward_start: Start date of underlying swap
            projection_curve: Curve for rate calculation
        """
        # Forward swap rate = (DF_start - DF_end) / Annuity
        t_start = projection_curve.time_to_maturity(forward_start)
        t_end = t_start + self._cms_tenor_years

        if t_start <= 0:
            t_start = 0.001

        df_start = projection_curve.discount_factor(t_start)
        df_end = projection_curve.discount_factor(t_end)

        # Calculate annuity (simplified: assume semi-annual payments)
        annuity = 0.0
        t = t_start + 0.5
        while t <= t_end:
            annuity += 0.5 * projection_curve.discount_factor(t)
            t += 0.5

        if annuity < 1e-10:
            return 0.0

        return (df_start - df_end) / annuity

    def _cms_convexity_adjustment(
        self,
        forward_rate: float,
        time_to_fixing: float,
        volatility: float,
        mean_reversion: float = 0.03,
    ) -> float:
        """
        Calculate CMS convexity adjustment using the Linear Swap Rate (LSR) model.

        The convexity adjustment arises because:
        - CMS pays the swap rate at a single date
        - Standard swap pays rate * accrual over multiple dates
        - The timing mismatch creates a convexity effect

        Formula (Hagan's approximation):
        CA ≈ σ² × T × S × G'(S) / G(S)

        where:
            S = forward swap rate
            σ = volatility
            T = time to fixing
            G(S) = PV01/N = annuity per unit notional
            G'(S) = dG/dS = derivative of annuity w.r.t. swap rate

        Simplified approximation using swap duration:
        CA ≈ σ² × T × S × Duration² / (1 + S × Duration)

        Args:
            forward_rate: Forward swap rate
            time_to_fixing: Time to CMS fixing
            volatility: Swaption (lognormal) volatility
            mean_reversion: Mean reversion parameter (not used in this approximation)
        """
        if time_to_fixing <= 0 or volatility <= 0 or forward_rate <= 0:
            return 0.0

        # Calculate Macaulay duration of underlying swap
        # For a par swap paying semi-annually:
        # Duration ≈ (1 - (1+S/m)^(-n)) / S
        # where m = payments per year, n = total years

        m = 2  # Assume semi-annual for underlying swap
        n = self._cms_tenor_years

        # Modified duration approximation
        if forward_rate > 0:
            # Annuity factor for semi-annual payments
            periodic_rate = forward_rate / m
            num_periods = n * m

            if periodic_rate > 1e-10:
                # PV of annuity
                annuity_factor = (1 - (1 + periodic_rate) ** (-num_periods)) / periodic_rate
                # Macaulay duration
                duration = annuity_factor / m
            else:
                duration = n
        else:
            duration = n

        # Convexity of the annuity (second derivative effect)
        # d(Duration)/dS ≈ -Duration² / (1 + S * Duration/m)
        duration_sensitivity = duration * duration / (1 + forward_rate * duration)

        # CMS Convexity Adjustment (Hagan's linear approximation)
        # CA = σ² × T × S × |G'(S)| / G(S)
        # where G'(S)/G(S) ≈ -Duration / (1 + S × Duration)
        vol_squared = volatility ** 2
        convexity_adj = vol_squared * time_to_fixing * forward_rate * duration_sensitivity

        return convexity_adj

    def fixed_leg_pv(
        self,
        valuation_date: date,
        discount_curve: Curve,
    ) -> float:
        """Calculate fixed leg present value."""
        pv = 0.0

        for accrual_start, accrual_end in self._fixed_schedule.periods:
            payment_date = self._get_payment_date(accrual_end)

            if payment_date < valuation_date:
                continue

            dcf = self.fixed_day_count.year_fraction(accrual_start, accrual_end)
            df = discount_curve.discount_factor(payment_date)

            pv += self.notional * self.fixed_rate * dcf * df

        return pv

    def float_leg_pv(
        self,
        valuation_date: date,
        projection_curve: Curve,
        discount_curve: Optional[Curve] = None,
        volatility: float = 0.20,
        mean_reversion: float = 0.03,
    ) -> float:
        """
        Calculate CMS floating leg present value.

        Args:
            valuation_date: Valuation date
            projection_curve: Curve for projecting forward swap rates
            discount_curve: Discount curve
            volatility: Swaption volatility for convexity adjustment
            mean_reversion: Mean reversion for convexity adjustment
        """
        if discount_curve is None:
            discount_curve = projection_curve

        pv = 0.0

        for accrual_start, accrual_end in self._float_schedule.periods:
            payment_date = self._get_payment_date(accrual_end)

            if payment_date < valuation_date:
                continue

            # Forward swap rate
            forward_rate = self._forward_swap_rate(accrual_start, projection_curve)

            # Time to fixing
            time_to_fixing = (accrual_start - valuation_date).days / 365.0

            # Convexity adjustment
            convexity_adj = self._cms_convexity_adjustment(
                forward_rate, time_to_fixing, volatility, mean_reversion
            )

            # Adjusted CMS rate
            cms_rate = forward_rate + convexity_adj + self.spread

            dcf = self.float_day_count.year_fraction(accrual_start, accrual_end)
            df = discount_curve.discount_factor(payment_date)

            pv += self.notional * cms_rate * dcf * df

        return pv

    def present_value(
        self,
        valuation_date: date,
        projection_curve: Curve,
        discount_curve: Optional[Curve] = None,
        volatility: float = 0.20,
        mean_reversion: float = 0.03,
    ) -> float:
        """Calculate swap NPV."""
        if discount_curve is None:
            discount_curve = projection_curve

        fixed_pv = self.fixed_leg_pv(valuation_date, discount_curve)
        float_pv = self.float_leg_pv(
            valuation_date, projection_curve, discount_curve, volatility, mean_reversion
        )

        if self.pay_fixed:
            return float_pv - fixed_pv
        else:
            return fixed_pv - float_pv

    def par_rate(
        self,
        valuation_date: date,
        projection_curve: Curve,
        discount_curve: Optional[Curve] = None,
        volatility: float = 0.20,
        mean_reversion: float = 0.03,
    ) -> float:
        """Calculate par CMS spread."""
        if discount_curve is None:
            discount_curve = projection_curve

        # CMS leg PV with zero spread
        original_spread = self.spread
        self.spread = 0.0
        cms_pv = self.float_leg_pv(
            valuation_date, projection_curve, discount_curve, volatility, mean_reversion
        )
        self.spread = original_spread

        # Calculate annuity
        annuity = 0.0
        for accrual_start, accrual_end in self._fixed_schedule.periods:
            payment_date = self._get_payment_date(accrual_end)

            if payment_date < valuation_date:
                continue

            dcf = self.fixed_day_count.year_fraction(accrual_start, accrual_end)
            df = discount_curve.discount_factor(payment_date)

            annuity += dcf * df

        if abs(annuity) < 1e-10:
            return 0.0

        return cms_pv / (self.notional * annuity)

    def __repr__(self) -> str:
        swap_type = "Payer" if self.pay_fixed else "Receiver"
        return (
            f"CMSSwap({swap_type}, "
            f"notional={self.notional:,.0f}, "
            f"rate={self.fixed_rate:.4%}, "
            f"CMS={self.cms_tenor}, "
            f"{self.start_date} to {self.end_date})"
        )


# =============================================================================
# Callable/Puttable Swap
# =============================================================================

class CallablePuttableSwap:
    """
    Swap with embedded Bermudan exercise option.

    - Callable: Issuer (fixed payer) can terminate early
    - Puttable: Holder (fixed receiver) can terminate early

    This is equivalent to a swap plus a Bermudan swaption.
    """

    def __init__(
        self,
        notional: float,
        start_date: date,
        end_date: date,
        fixed_rate: float,
        pay_fixed: bool = True,
        is_callable: bool = True,
        exercise_schedule: Optional[ExerciseSchedule] = None,
        index_tenor: str = "3M",
        spread: float = 0.0,
        fixed_frequency: Union[str, Frequency] = Frequency.SEMI_ANNUAL,
        float_frequency: Union[str, Frequency] = Frequency.QUARTERLY,
        fixed_day_count: Union[str, DayCountConvention] = "30/360",
        float_day_count: Union[str, DayCountConvention] = "ACT/360",
        calendar: Optional[Calendar] = None,
    ):
        """
        Initialize callable/puttable swap.

        Args:
            notional: Notional principal
            start_date: Effective date
            end_date: Maturity date
            fixed_rate: Fixed leg rate
            pay_fixed: True = payer swap
            is_callable: True = callable (exercisable by payer), False = puttable
            exercise_schedule: Bermudan exercise dates
            index_tenor: Floating rate index tenor
            spread: Spread over floating rate
            fixed_frequency: Fixed leg payment frequency
            float_frequency: Float leg payment frequency
            fixed_day_count: Fixed leg day count
            float_day_count: Float leg day count
            calendar: Business day calendar
        """
        self.notional = notional
        self.start_date = start_date
        self.end_date = end_date
        self.fixed_rate = fixed_rate
        self.pay_fixed = pay_fixed
        self.is_callable = is_callable
        self.index_tenor = index_tenor
        self.spread = spread

        # Day counts
        if isinstance(fixed_day_count, str):
            fixed_day_count = _get_convention(fixed_day_count)
        if isinstance(float_day_count, str):
            float_day_count = _get_convention(float_day_count)

        self.fixed_day_count = fixed_day_count
        self.float_day_count = float_day_count

        # Parse frequencies
        if isinstance(fixed_frequency, str):
            fixed_frequency = Frequency[fixed_frequency.upper()]
        if isinstance(float_frequency, str):
            float_frequency = Frequency[float_frequency.upper()]

        self.fixed_frequency = fixed_frequency
        self.float_frequency = float_frequency

        self.calendar = calendar

        # Exercise schedule
        if exercise_schedule is None:
            # Default: annual exercise on fixed payment dates
            exercise_dates = []
            schedule = Schedule(
                effective_date=start_date,
                termination_date=end_date,
                frequency=fixed_frequency,
                calendar=calendar,
            )
            for _, accrual_end in schedule.periods[:-1]:  # Exclude final date
                exercise_dates.append(accrual_end)

            self.exercise_schedule = ExerciseSchedule(
                exercise_dates=exercise_dates,
                exercise_type=ExerciseType.BERMUDAN,
            )
        else:
            self.exercise_schedule = exercise_schedule

        # Parse index tenor
        self._index_tenor_years = self._parse_tenor(index_tenor)

        # Generate schedules
        self._fixed_schedule = Schedule(
            effective_date=start_date,
            termination_date=end_date,
            frequency=fixed_frequency,
            calendar=calendar,
        )

        self._float_schedule = Schedule(
            effective_date=start_date,
            termination_date=end_date,
            frequency=float_frequency,
            calendar=calendar,
        )

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
        return 0.25

    @property
    def is_puttable(self) -> bool:
        """Check if swap is puttable (exercisable by receiver)."""
        return not self.is_callable

    def _get_payment_date(self, accrual_end: date) -> date:
        """Get payment date from accrual end."""
        if self.calendar:
            adjustment = BusinessDayAdjustment("modified_following", self.calendar)
            return adjustment.adjust(accrual_end)
        return accrual_end

    def _project_rate(
        self,
        accrual_start: date,
        accrual_end: date,
        projection_curve: Curve,
    ) -> float:
        """Project forward rate from curve."""
        t1 = projection_curve.time_to_maturity(accrual_start)
        t2 = projection_curve.time_to_maturity(accrual_end)

        if t1 <= 0:
            t1 = 0.001
        if t2 <= t1:
            t2 = t1 + self._index_tenor_years

        return projection_curve.forward_rate(t1, t2)

    def _underlying_swap_pv(
        self,
        valuation_date: date,
        exercise_date: date,
        projection_curve: Curve,
        discount_curve: Curve,
    ) -> float:
        """
        Calculate PV of underlying swap starting at exercise date.
        """
        pv = 0.0

        # Fixed leg
        for accrual_start, accrual_end in self._fixed_schedule.periods:
            if accrual_start < exercise_date:
                continue

            payment_date = self._get_payment_date(accrual_end)
            dcf = self.fixed_day_count.year_fraction(accrual_start, accrual_end)
            df = discount_curve.discount_factor(payment_date)

            pv -= self.notional * self.fixed_rate * dcf * df

        # Floating leg
        for accrual_start, accrual_end in self._float_schedule.periods:
            if accrual_start < exercise_date:
                continue

            payment_date = self._get_payment_date(accrual_end)
            rate = self._project_rate(accrual_start, accrual_end, projection_curve)
            dcf = self.float_day_count.year_fraction(accrual_start, accrual_end)
            df = discount_curve.discount_factor(payment_date)

            pv += self.notional * (rate + self.spread) * dcf * df

        return pv

    def _bermudan_swaption_value(
        self,
        valuation_date: date,
        projection_curve: Curve,
        discount_curve: Curve,
        volatility: float = 0.20,
    ) -> float:
        """
        Estimate Bermudan swaption value using backward induction.

        This is a simplified valuation. Full implementation would use
        a term structure model (e.g., LGM, Hull-White) with numerical methods.
        """
        if not self.exercise_schedule.exercise_dates:
            return 0.0

        # Get future exercise dates
        future_exercises = [
            d for d in self.exercise_schedule.exercise_dates
            if d > valuation_date
        ]

        if not future_exercises:
            return 0.0

        # Simplified: Use maximum of European swaption values
        # (Lower bound for Bermudan)
        max_value = 0.0

        for exercise_date in future_exercises:
            # Forward swap value at exercise
            swap_value = self._underlying_swap_pv(
                valuation_date, exercise_date, projection_curve, discount_curve
            )

            # Time to exercise
            time_to_exercise = (exercise_date - valuation_date).days / 365.0

            if time_to_exercise <= 0:
                exercise_value = max(0, swap_value) if self.is_callable else max(0, -swap_value)
            else:
                # Simple Black approximation for European swaption
                # More sophisticated: use forward swap rate and annuity
                remaining_maturity = (self.end_date - exercise_date).days / 365.0

                if remaining_maturity > 0:
                    # Approximate forward swap rate
                    t_ex = projection_curve.time_to_maturity(exercise_date)
                    t_end = projection_curve.time_to_maturity(self.end_date)

                    df_ex = projection_curve.discount_factor(t_ex)
                    df_end = projection_curve.discount_factor(t_end)

                    # Simplified annuity
                    annuity = remaining_maturity * df_end

                    if annuity > 0:
                        forward_rate = (df_ex - df_end) / annuity
                    else:
                        forward_rate = self.fixed_rate

                    # Black's formula approximation
                    sqrt_t = math.sqrt(time_to_exercise)
                    vol_sqrt_t = volatility * sqrt_t

                    if vol_sqrt_t > 0 and forward_rate > 0 and self.fixed_rate > 0:
                        d1 = (math.log(forward_rate / self.fixed_rate) + 0.5 * volatility ** 2 * time_to_exercise) / vol_sqrt_t
                        d2 = d1 - vol_sqrt_t

                        # Normal CDF approximation
                        def norm_cdf(x):
                            a1 = 0.254829592
                            a2 = -0.284496736
                            a3 = 1.421413741
                            a4 = -1.453152027
                            a5 = 1.061405429
                            p = 0.3275911

                            sign = 1 if x >= 0 else -1
                            x = abs(x) / math.sqrt(2)

                            t = 1.0 / (1.0 + p * x)
                            y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

                            return 0.5 * (1.0 + sign * y)

                        if self.is_callable:
                            # Receiver swaption (right to receive fixed)
                            exercise_value = self.notional * annuity * (
                                self.fixed_rate * norm_cdf(-d2) - forward_rate * norm_cdf(-d1)
                            )
                        else:
                            # Payer swaption (right to pay fixed)
                            exercise_value = self.notional * annuity * (
                                forward_rate * norm_cdf(d1) - self.fixed_rate * norm_cdf(d2)
                            )

                        exercise_value = max(0, exercise_value)
                    else:
                        exercise_value = 0.0
                else:
                    exercise_value = 0.0

            max_value = max(max_value, exercise_value)

        # Bermudan premium (typically 10-20% over max European)
        # This is a rough approximation
        bermudan_premium = 1.15 if len(future_exercises) > 1 else 1.0

        return max_value * bermudan_premium

    def underlying_swap_pv(
        self,
        valuation_date: date,
        projection_curve: Curve,
        discount_curve: Optional[Curve] = None,
    ) -> float:
        """Calculate PV of underlying (non-callable) swap."""
        if discount_curve is None:
            discount_curve = projection_curve

        # Fixed leg PV
        fixed_pv = 0.0
        for accrual_start, accrual_end in self._fixed_schedule.periods:
            payment_date = self._get_payment_date(accrual_end)

            if payment_date < valuation_date:
                continue

            dcf = self.fixed_day_count.year_fraction(accrual_start, accrual_end)
            df = discount_curve.discount_factor(payment_date)

            fixed_pv += self.notional * self.fixed_rate * dcf * df

        # Floating leg PV
        float_pv = 0.0
        for accrual_start, accrual_end in self._float_schedule.periods:
            payment_date = self._get_payment_date(accrual_end)

            if payment_date < valuation_date:
                continue

            rate = self._project_rate(accrual_start, accrual_end, projection_curve)
            dcf = self.float_day_count.year_fraction(accrual_start, accrual_end)
            df = discount_curve.discount_factor(payment_date)

            float_pv += self.notional * (rate + self.spread) * dcf * df

        if self.pay_fixed:
            return float_pv - fixed_pv
        else:
            return fixed_pv - float_pv

    def option_value(
        self,
        valuation_date: date,
        projection_curve: Curve,
        discount_curve: Optional[Curve] = None,
        volatility: float = 0.20,
    ) -> float:
        """Calculate value of embedded option."""
        if discount_curve is None:
            discount_curve = projection_curve

        return self._bermudan_swaption_value(
            valuation_date, projection_curve, discount_curve, volatility
        )

    def present_value(
        self,
        valuation_date: date,
        projection_curve: Curve,
        discount_curve: Optional[Curve] = None,
        volatility: float = 0.20,
    ) -> float:
        """
        Calculate callable/puttable swap NPV.

        Value = Underlying Swap +/- Option Value

        For callable (exercisable by payer):
            Payer's value = Swap PV - Receiver Swaption Value

        For puttable (exercisable by receiver):
            Receiver's value = Swap PV + Payer Swaption Value
        """
        if discount_curve is None:
            discount_curve = projection_curve

        swap_pv = self.underlying_swap_pv(valuation_date, projection_curve, discount_curve)
        option_pv = self.option_value(valuation_date, projection_curve, discount_curve, volatility)

        if self.is_callable:
            # Callable: payer has given away option to receiver
            if self.pay_fixed:
                return swap_pv - option_pv
            else:
                return swap_pv + option_pv
        else:
            # Puttable: receiver has the option
            if self.pay_fixed:
                return swap_pv + option_pv
            else:
                return swap_pv - option_pv

    def __repr__(self) -> str:
        swap_type = "Payer" if self.pay_fixed else "Receiver"
        option_type = "Callable" if self.is_callable else "Puttable"
        return (
            f"{option_type}Swap({swap_type}, "
            f"notional={self.notional:,.0f}, "
            f"rate={self.fixed_rate:.4%}, "
            f"exercises={len(self.exercise_schedule.exercise_dates)}, "
            f"{self.start_date} to {self.end_date})"
        )


# Aliases for clarity
CallableSwap = CallablePuttableSwap
PuttableSwap = CallablePuttableSwap


# =============================================================================
# Factory functions
# =============================================================================

def create_sofr_swap(
    notional: float,
    start_date: date,
    end_date: date,
    fixed_rate: float,
    pay_fixed: bool = True,
    spread: float = 0.0,
    lookback_days: int = 2,
) -> CompoundedRFRSwap:
    """
    Create a standard SOFR swap with ISDA conventions.

    Args:
        notional: Notional principal
        start_date: Effective date
        end_date: Maturity date
        fixed_rate: Fixed leg rate
        pay_fixed: True = payer
        spread: Spread on floating leg
        lookback_days: Observation lookback (default 2 for SOFR)
    """
    return CompoundedRFRSwap(
        notional=notional,
        start_date=start_date,
        end_date=end_date,
        fixed_rate=fixed_rate,
        pay_fixed=pay_fixed,
        spread=spread,
        compounding_method=RFRCompoundingMethod.COMPOUNDED,
        observation_shift=ObservationShift.NONE,
        lookback_days=lookback_days,
        lockout_days=0,
        payment_delay=2,
        fixed_frequency=Frequency.ANNUAL,
        float_frequency=Frequency.ANNUAL,
        fixed_day_count="ACT/360",
        float_day_count="ACT/360",
        index_name="SOFR",
    )


def create_sonia_swap(
    notional: float,
    start_date: date,
    end_date: date,
    fixed_rate: float,
    pay_fixed: bool = True,
    spread: float = 0.0,
) -> CompoundedRFRSwap:
    """
    Create a standard SONIA swap with ISDA conventions.

    SONIA uses 5 business day observation shift (shifted lookback).
    """
    return CompoundedRFRSwap(
        notional=notional,
        start_date=start_date,
        end_date=end_date,
        fixed_rate=fixed_rate,
        pay_fixed=pay_fixed,
        spread=spread,
        compounding_method=RFRCompoundingMethod.COMPOUNDED,
        observation_shift=ObservationShift.OBSERVATION,
        lookback_days=5,
        lockout_days=0,
        payment_delay=0,
        fixed_frequency=Frequency.ANNUAL,
        float_frequency=Frequency.ANNUAL,
        fixed_day_count="ACT/365",
        float_day_count="ACT/365",
        index_name="SONIA",
    )


def create_estr_swap(
    notional: float,
    start_date: date,
    end_date: date,
    fixed_rate: float,
    pay_fixed: bool = True,
    spread: float = 0.0,
) -> CompoundedRFRSwap:
    """
    Create a standard ESTR (€STR) swap with ISDA conventions.

    ESTR uses 2 business day payment delay.
    """
    return CompoundedRFRSwap(
        notional=notional,
        start_date=start_date,
        end_date=end_date,
        fixed_rate=fixed_rate,
        pay_fixed=pay_fixed,
        spread=spread,
        compounding_method=RFRCompoundingMethod.COMPOUNDED,
        observation_shift=ObservationShift.NONE,
        lookback_days=0,
        lockout_days=0,
        payment_delay=2,
        fixed_frequency=Frequency.ANNUAL,
        float_frequency=Frequency.ANNUAL,
        fixed_day_count="ACT/360",
        float_day_count="ACT/360",
        index_name="ESTR",
    )
