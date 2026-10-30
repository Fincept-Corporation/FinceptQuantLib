"""
Cross-Currency Swaps

Implements:
- Cross-currency interest rate swaps (XCCY)
- Fixed-floating cross-currency swaps
- Floating-floating cross-currency basis swaps

A cross-currency swap involves:
1. Initial exchange of principals in different currencies
2. Periodic interest payments in each currency
3. Final re-exchange of principals

Key considerations:
- Currency basis spread
- FX exposure
- Dual discounting (OIS curves for each currency)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Union, Optional
from finceptquantlib.curves.curve import Curve
from finceptquantlib.scheduling.schedule import Schedule, Frequency
from finceptquantlib.scheduling.daycount import DayCountConvention, _get_convention


@dataclass
class XCCYResult:
    """Result of cross-currency swap valuation."""
    present_value: float          # PV in domestic currency
    pv_domestic_leg: float        # PV of domestic leg
    pv_foreign_leg: float         # PV of foreign leg (converted)
    par_spread: float             # Par basis spread
    fx_delta: float               # FX rate sensitivity
    domestic_ir_delta: float      # Domestic rate sensitivity
    foreign_ir_delta: float       # Foreign rate sensitivity


class CrossCurrencySwap:
    """
    Cross-Currency Interest Rate Swap.

    Exchange of interest payments in two different currencies,
    with exchange of notionals at start and end.

    Common types:
    - Fixed-Fixed: Exchange fixed rates in both currencies
    - Fixed-Floating: Exchange fixed in one currency for floating in another
    - Floating-Floating (Basis): Exchange floating rates (e.g., USD LIBOR vs EUR LIBOR)
    """

    def __init__(
        self,
        domestic_currency: str,
        foreign_currency: str,
        domestic_notional: float,
        foreign_notional: float,
        start_date: date,
        end_date: date,
        domestic_fixed_rate: Optional[float] = None,
        foreign_fixed_rate: Optional[float] = None,
        domestic_spread: float = 0.0,
        foreign_spread: float = 0.0,
        domestic_frequency: Union[str, Frequency] = Frequency.QUARTERLY,
        foreign_frequency: Union[str, Frequency] = Frequency.QUARTERLY,
        domestic_day_count: Union[str, DayCountConvention] = "ACT/360",
        foreign_day_count: Union[str, DayCountConvention] = "ACT/360",
        exchange_initial: bool = True,
        exchange_final: bool = True,
        is_receive_domestic: bool = True,
    ):
        """
        Initialize cross-currency swap.

        Args:
            domestic_currency: Domestic (quote) currency
            foreign_currency: Foreign (base) currency
            domestic_notional: Notional in domestic currency
            foreign_notional: Notional in foreign currency
            start_date: Swap effective date
            end_date: Swap maturity date
            domestic_fixed_rate: Fixed rate for domestic leg (None for floating)
            foreign_fixed_rate: Fixed rate for foreign leg (None for floating)
            domestic_spread: Spread on domestic floating leg
            foreign_spread: Spread on foreign floating leg
            domestic_frequency: Domestic payment frequency
            foreign_frequency: Foreign payment frequency
            domestic_day_count: Domestic day count convention
            foreign_day_count: Foreign day count convention
            exchange_initial: Exchange notionals at start
            exchange_final: Exchange notionals at end
            is_receive_domestic: True = receive domestic, pay foreign
        """
        self.domestic_currency = domestic_currency.upper()
        self.foreign_currency = foreign_currency.upper()
        self.domestic_notional = domestic_notional
        self.foreign_notional = foreign_notional
        self.start_date = start_date
        self.end_date = end_date
        self.domestic_fixed_rate = domestic_fixed_rate
        self.foreign_fixed_rate = foreign_fixed_rate
        self.domestic_spread = domestic_spread
        self.foreign_spread = foreign_spread
        self.exchange_initial = exchange_initial
        self.exchange_final = exchange_final
        self.is_receive_domestic = is_receive_domestic

        # Schedules
        if isinstance(domestic_frequency, str):
            domestic_frequency = Frequency[domestic_frequency.upper()]
        if isinstance(foreign_frequency, str):
            foreign_frequency = Frequency[foreign_frequency.upper()]

        self.domestic_frequency = domestic_frequency
        self.foreign_frequency = foreign_frequency

        if isinstance(domestic_day_count, str):
            domestic_day_count = _get_convention(domestic_day_count)
        if isinstance(foreign_day_count, str):
            foreign_day_count = _get_convention(foreign_day_count)

        self.domestic_day_count = domestic_day_count
        self.foreign_day_count = foreign_day_count

        self._domestic_schedule = Schedule(
            effective_date=start_date,
            termination_date=end_date,
            frequency=domestic_frequency,
        )

        self._foreign_schedule = Schedule(
            effective_date=start_date,
            termination_date=end_date,
            frequency=foreign_frequency,
        )

    @property
    def implied_fx_rate(self) -> float:
        """Implied FX rate from notionals."""
        return self.domestic_notional / self.foreign_notional

    def _project_rate(
        self,
        curve: Curve,
        accrual_start: date,
        accrual_end: date,
    ) -> float:
        """Project floating rate from curve."""
        t1 = curve.time_to_maturity(accrual_start)
        t2 = curve.time_to_maturity(accrual_end)
        if t1 <= 0:
            t1 = 0.001
        if t2 <= t1:
            t2 = t1 + 0.25
        return curve.forward_rate(t1, t2)

    def domestic_leg_pv(
        self,
        valuation_date: date,
        domestic_discount_curve: Curve,
        domestic_projection_curve: Optional[Curve] = None,
    ) -> float:
        """
        Calculate present value of domestic leg.

        Returns PV in domestic currency.
        """
        if domestic_projection_curve is None:
            domestic_projection_curve = domestic_discount_curve

        total = 0.0

        # Initial exchange (pay domestic notional)
        if self.exchange_initial and self.start_date > valuation_date:
            df = domestic_discount_curve.discount_factor(self.start_date)
            total -= self.domestic_notional * df

        # Interest payments
        for accrual_start, accrual_end in self._domestic_schedule.periods:
            if accrual_end <= valuation_date:
                continue

            dcf = self.domestic_day_count.year_fraction(accrual_start, accrual_end)
            df = domestic_discount_curve.discount_factor(accrual_end)

            if self.domestic_fixed_rate is not None:
                rate = self.domestic_fixed_rate
            else:
                rate = self._project_rate(
                    domestic_projection_curve, accrual_start, accrual_end
                ) + self.domestic_spread

            total += self.domestic_notional * rate * dcf * df

        # Final exchange (receive domestic notional)
        if self.exchange_final and self.end_date > valuation_date:
            df = domestic_discount_curve.discount_factor(self.end_date)
            total += self.domestic_notional * df

        return total

    def foreign_leg_pv(
        self,
        valuation_date: date,
        foreign_discount_curve: Curve,
        foreign_projection_curve: Optional[Curve] = None,
    ) -> float:
        """
        Calculate present value of foreign leg.

        Returns PV in foreign currency.
        """
        if foreign_projection_curve is None:
            foreign_projection_curve = foreign_discount_curve

        total = 0.0

        # Initial exchange (receive foreign notional)
        if self.exchange_initial and self.start_date > valuation_date:
            df = foreign_discount_curve.discount_factor(self.start_date)
            total += self.foreign_notional * df

        # Interest payments
        for accrual_start, accrual_end in self._foreign_schedule.periods:
            if accrual_end <= valuation_date:
                continue

            dcf = self.foreign_day_count.year_fraction(accrual_start, accrual_end)
            df = foreign_discount_curve.discount_factor(accrual_end)

            if self.foreign_fixed_rate is not None:
                rate = self.foreign_fixed_rate
            else:
                rate = self._project_rate(
                    foreign_projection_curve, accrual_start, accrual_end
                ) + self.foreign_spread

            total -= self.foreign_notional * rate * dcf * df

        # Final exchange (pay foreign notional)
        if self.exchange_final and self.end_date > valuation_date:
            df = foreign_discount_curve.discount_factor(self.end_date)
            total -= self.foreign_notional * df

        return total

    def present_value(
        self,
        valuation_date: date,
        fx_rate: float,
        domestic_discount_curve: Curve,
        foreign_discount_curve: Curve,
        domestic_projection_curve: Optional[Curve] = None,
        foreign_projection_curve: Optional[Curve] = None,
    ) -> float:
        """
        Calculate total present value in domestic currency.

        Args:
            valuation_date: Valuation date
            fx_rate: Current FX rate (domestic per foreign)
            domestic_discount_curve: Domestic OIS curve
            foreign_discount_curve: Foreign OIS curve
            domestic_projection_curve: Domestic floating rate curve
            foreign_projection_curve: Foreign floating rate curve

        Returns:
            NPV in domestic currency
        """
        dom_pv = self.domestic_leg_pv(
            valuation_date, domestic_discount_curve, domestic_projection_curve
        )
        for_pv = self.foreign_leg_pv(
            valuation_date, foreign_discount_curve, foreign_projection_curve
        )

        # Convert foreign PV to domestic
        for_pv_domestic = for_pv * fx_rate

        if self.is_receive_domestic:
            return dom_pv + for_pv_domestic
        else:
            return -dom_pv - for_pv_domestic

    def par_spread(
        self,
        valuation_date: date,
        fx_rate: float,
        domestic_discount_curve: Curve,
        foreign_discount_curve: Curve,
        domestic_projection_curve: Optional[Curve] = None,
        foreign_projection_curve: Optional[Curve] = None,
        spread_leg: str = "foreign",
    ) -> float:
        """
        Calculate par (break-even) basis spread.

        The spread on one leg that makes the swap have zero NPV.

        Args:
            valuation_date: Valuation date
            fx_rate: Current FX rate
            domestic_discount_curve: Domestic curve
            foreign_discount_curve: Foreign curve
            domestic_projection_curve: Domestic projection
            foreign_projection_curve: Foreign projection
            spread_leg: Which leg to solve for spread ("domestic" or "foreign")

        Returns:
            Par basis spread
        """
        if spread_leg.lower() == "foreign":
            # Solve for foreign spread
            temp_swap = CrossCurrencySwap(
                domestic_currency=self.domestic_currency,
                foreign_currency=self.foreign_currency,
                domestic_notional=self.domestic_notional,
                foreign_notional=self.foreign_notional,
                start_date=self.start_date,
                end_date=self.end_date,
                domestic_fixed_rate=self.domestic_fixed_rate,
                foreign_fixed_rate=self.foreign_fixed_rate,
                domestic_spread=self.domestic_spread,
                foreign_spread=0.0,
                domestic_frequency=self.domestic_frequency,
                foreign_frequency=self.foreign_frequency,
                domestic_day_count=self.domestic_day_count,
                foreign_day_count=self.foreign_day_count,
                exchange_initial=self.exchange_initial,
                exchange_final=self.exchange_final,
                is_receive_domestic=self.is_receive_domestic,
            )

            # Calculate PV with zero spread
            npv_zero_spread = temp_swap.present_value(
                valuation_date, fx_rate,
                domestic_discount_curve, foreign_discount_curve,
                domestic_projection_curve, foreign_projection_curve
            )

            # Calculate annuity
            annuity = 0.0
            for accrual_start, accrual_end in self._foreign_schedule.periods:
                if accrual_end <= valuation_date:
                    continue
                dcf = self.foreign_day_count.year_fraction(accrual_start, accrual_end)
                df = foreign_discount_curve.discount_factor(accrual_end)
                annuity += dcf * df

            if abs(annuity) < 1e-10:
                return 0.0

            # Sign adjustment based on direction
            sign = 1 if self.is_receive_domestic else -1
            return sign * npv_zero_spread / (self.foreign_notional * annuity * fx_rate)

        else:
            # Similar calculation for domestic spread
            return 0.0  # Simplified

    def fx_delta(
        self,
        valuation_date: date,
        fx_rate: float,
        domestic_discount_curve: Curve,
        foreign_discount_curve: Curve,
        bump_pct: float = 1.0,
    ) -> float:
        """
        Calculate FX delta (sensitivity to FX rate change).

        Args:
            valuation_date: Valuation date
            fx_rate: Current FX rate
            domestic_discount_curve: Domestic curve
            foreign_discount_curve: Foreign curve
            bump_pct: Percentage bump to FX rate

        Returns:
            Change in NPV per 1% FX move
        """
        base_npv = self.present_value(
            valuation_date, fx_rate,
            domestic_discount_curve, foreign_discount_curve
        )

        bumped_fx = fx_rate * (1 + bump_pct / 100)
        bumped_npv = self.present_value(
            valuation_date, bumped_fx,
            domestic_discount_curve, foreign_discount_curve
        )

        return bumped_npv - base_npv

    def value(
        self,
        valuation_date: date,
        fx_rate: float,
        domestic_discount_curve: Curve,
        foreign_discount_curve: Curve,
        domestic_projection_curve: Optional[Curve] = None,
        foreign_projection_curve: Optional[Curve] = None,
    ) -> XCCYResult:
        """
        Full valuation with all analytics.

        Returns:
            XCCYResult with NPV and sensitivities
        """
        npv = self.present_value(
            valuation_date, fx_rate,
            domestic_discount_curve, foreign_discount_curve,
            domestic_projection_curve, foreign_projection_curve
        )

        dom_pv = self.domestic_leg_pv(
            valuation_date, domestic_discount_curve, domestic_projection_curve
        )
        for_pv = self.foreign_leg_pv(
            valuation_date, foreign_discount_curve, foreign_projection_curve
        )

        par = self.par_spread(
            valuation_date, fx_rate,
            domestic_discount_curve, foreign_discount_curve,
            domestic_projection_curve, foreign_projection_curve
        )

        fx_d = self.fx_delta(
            valuation_date, fx_rate,
            domestic_discount_curve, foreign_discount_curve
        )

        # IR deltas
        base_npv = npv

        dom_up = domestic_discount_curve.shift(1)
        npv_dom_up = self.present_value(
            valuation_date, fx_rate,
            dom_up, foreign_discount_curve,
            domestic_projection_curve, foreign_projection_curve
        )
        dom_ir_delta = npv_dom_up - base_npv

        for_up = foreign_discount_curve.shift(1)
        npv_for_up = self.present_value(
            valuation_date, fx_rate,
            domestic_discount_curve, for_up,
            domestic_projection_curve, foreign_projection_curve
        )
        for_ir_delta = npv_for_up - base_npv

        return XCCYResult(
            present_value=npv,
            pv_domestic_leg=dom_pv,
            pv_foreign_leg=for_pv * fx_rate,
            par_spread=par,
            fx_delta=fx_d,
            domestic_ir_delta=dom_ir_delta,
            foreign_ir_delta=for_ir_delta,
        )

    def __repr__(self) -> str:
        dom_type = "Fixed" if self.domestic_fixed_rate is not None else "Float"
        for_type = "Fixed" if self.foreign_fixed_rate is not None else "Float"
        return (f"CrossCurrencySwap({dom_type}/{for_type}, "
                f"{self.domestic_currency}/{self.foreign_currency}, "
                f"notional={self.domestic_notional:,.0f}/{self.foreign_notional:,.0f}, "
                f"{self.start_date} to {self.end_date})")
