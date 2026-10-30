"""
Cross-Currency Swaps

Provides sophisticated cross-currency swap structures:
- Standard Cross-Currency Swap: Principal exchange with interest
- MTM Cross-Currency Swap: Mark-to-market resets with FX adjustments
- Resetting Cross-Currency Swap: Periodic notional resets
- Cross-Currency Basis Swap: Float vs Float with different currencies

Key Features:
- Multi-curve framework with separate discount/projection curves per currency
- FX forward integration
- MTM notional reset mechanism
- Automatic triangulation for exotic currency pairs
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Union, Optional, List, Dict

from finceptquantlib.curves.curve import Curve
from finceptquantlib.scheduling.schedule import Schedule, Frequency
from finceptquantlib.scheduling.calendar import Calendar
from finceptquantlib.scheduling.daycount import DayCountConvention, _get_convention
from finceptquantlib.scheduling.adjustment import BusinessDayAdjustment


class XCCYSwapType(Enum):
    """Type of cross-currency swap."""
    FIXED_FIXED = "fixed_fixed"         # Fixed vs Fixed
    FIXED_FLOAT = "fixed_float"         # Fixed vs Float (most common)
    FLOAT_FLOAT = "float_float"         # Float vs Float (basis swap)


class ResetType(Enum):
    """MTM reset type."""
    NONE = "none"                       # Standard (no reset)
    FULL = "full"                       # Full notional reset
    PARTIAL = "partial"                 # Partial reset


@dataclass
class FXRate:
    """FX rate with bid/ask spread."""
    mid: float
    bid: Optional[float] = None
    ask: Optional[float] = None

    def __post_init__(self):
        if self.bid is None:
            self.bid = self.mid
        if self.ask is None:
            self.ask = self.mid

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask - self.bid


@dataclass
class XCCYCashFlow:
    """Cross-currency swap cash flow."""
    payment_date: date
    currency: str
    amount: float
    flow_type: str  # "interest", "principal", "mtm_adjustment"
    accrual_start: Optional[date] = None
    accrual_end: Optional[date] = None
    notional: Optional[float] = None
    rate: Optional[float] = None

    def discount(self, curve: Curve) -> float:
        """Discount to curve reference date."""
        t = curve.time_to_maturity(self.payment_date)
        if t <= 0:
            return self.amount
        return self.amount * curve.discount_factor(t)


@dataclass
class XCCYSwapResult:
    """Cross-currency swap valuation result."""
    npv_domestic: float
    npv_foreign: float
    npv_total: float  # In domestic currency
    domestic_leg_pv: float
    foreign_leg_pv: float
    fx_delta: float
    ir_delta_domestic: float
    ir_delta_foreign: float
    cash_flows_domestic: List[XCCYCashFlow]
    cash_flows_foreign: List[XCCYCashFlow]


class CrossCurrencySwap:
    """
    Standard Cross-Currency Swap.

    Exchange of notional and interest payments in two currencies.
    At inception: exchange notional at spot FX
    At maturity: exchange notional at original spot FX
    During life: exchange interest payments

    Example:
        - Domestic: USD fixed 2%
        - Foreign: EUR floating EURIBOR 3M
        - Spot: 1.10 EUR/USD
        - At start: Pay 100M USD, receive 90.91M EUR
        - At maturity: Receive 100M USD, pay 90.91M EUR
        - During: Pay USD fixed coupons, receive EUR floating coupons
    """

    def __init__(
        self,
        domestic_notional: float,
        domestic_currency: str,
        foreign_currency: str,
        start_date: date,
        end_date: date,
        spot_fx: float,  # Foreign per Domestic (e.g., 0.91 EUR/USD)
        swap_type: XCCYSwapType = XCCYSwapType.FIXED_FLOAT,
        # Domestic leg (typically fixed)
        domestic_rate: float = 0.02,
        domestic_is_fixed: bool = True,
        domestic_frequency: Union[str, Frequency] = Frequency.SEMI_ANNUAL,
        domestic_day_count: Union[str, DayCountConvention] = "30/360",
        domestic_spread: float = 0.0,
        domestic_index_tenor: str = "3M",
        # Foreign leg (typically floating)
        foreign_rate: float = 0.0,
        foreign_is_fixed: bool = False,
        foreign_frequency: Union[str, Frequency] = Frequency.QUARTERLY,
        foreign_day_count: Union[str, DayCountConvention] = "ACT/360",
        foreign_spread: float = 0.0,
        foreign_index_tenor: str = "3M",
        # Principal exchange
        exchange_initial_principal: bool = True,
        exchange_final_principal: bool = True,
        # Other settings
        calendar_domestic: Optional[Calendar] = None,
        calendar_foreign: Optional[Calendar] = None,
    ):
        """
        Initialize cross-currency swap.

        Args:
            domestic_notional: Notional in domestic currency
            domestic_currency: Domestic currency code (e.g., "USD")
            foreign_currency: Foreign currency code (e.g., "EUR")
            start_date: Effective date
            end_date: Maturity date
            spot_fx: Spot FX rate (foreign per domestic)
            swap_type: Type of XCCY swap
            domestic_rate: Fixed rate for domestic leg (if fixed)
            domestic_is_fixed: True if domestic leg is fixed
            domestic_frequency: Payment frequency for domestic leg
            domestic_day_count: Day count for domestic leg
            domestic_spread: Spread for domestic floating leg
            domestic_index_tenor: Index tenor for domestic floating leg
            foreign_rate: Fixed rate for foreign leg (if fixed)
            foreign_is_fixed: True if foreign leg is fixed
            foreign_frequency: Payment frequency for foreign leg
            foreign_day_count: Day count for foreign leg
            foreign_spread: Spread for foreign floating leg (basis)
            foreign_index_tenor: Index tenor for foreign floating leg
            exchange_initial_principal: Exchange notional at start
            exchange_final_principal: Exchange notional at maturity
            calendar_domestic: Calendar for domestic leg
            calendar_foreign: Calendar for foreign leg
        """
        self.domestic_notional = domestic_notional
        self.domestic_currency = domestic_currency.upper()
        self.foreign_currency = foreign_currency.upper()
        self.start_date = start_date
        self.end_date = end_date
        self.spot_fx = spot_fx
        self.swap_type = swap_type

        # Calculate foreign notional
        self.foreign_notional = domestic_notional * spot_fx

        # Domestic leg settings
        self.domestic_rate = domestic_rate
        self.domestic_is_fixed = domestic_is_fixed
        self.domestic_spread = domestic_spread
        self.domestic_index_tenor = domestic_index_tenor

        # Foreign leg settings
        self.foreign_rate = foreign_rate
        self.foreign_is_fixed = foreign_is_fixed
        self.foreign_spread = foreign_spread
        self.foreign_index_tenor = foreign_index_tenor

        # Principal exchange
        self.exchange_initial_principal = exchange_initial_principal
        self.exchange_final_principal = exchange_final_principal

        # Day counts
        if isinstance(domestic_day_count, str):
            domestic_day_count = _get_convention(domestic_day_count)
        if isinstance(foreign_day_count, str):
            foreign_day_count = _get_convention(foreign_day_count)

        self.domestic_day_count = domestic_day_count
        self.foreign_day_count = foreign_day_count

        # Parse frequencies
        if isinstance(domestic_frequency, str):
            domestic_frequency = Frequency[domestic_frequency.upper()]
        if isinstance(foreign_frequency, str):
            foreign_frequency = Frequency[foreign_frequency.upper()]

        self.domestic_frequency = domestic_frequency
        self.foreign_frequency = foreign_frequency

        self.calendar_domestic = calendar_domestic
        self.calendar_foreign = calendar_foreign

        # Parse index tenors
        self._domestic_tenor_years = self._parse_tenor(domestic_index_tenor)
        self._foreign_tenor_years = self._parse_tenor(foreign_index_tenor)

        # Generate schedules
        self._domestic_schedule = Schedule(
            effective_date=start_date,
            termination_date=end_date,
            frequency=domestic_frequency,
            calendar=calendar_domestic,
        )

        self._foreign_schedule = Schedule(
            effective_date=start_date,
            termination_date=end_date,
            frequency=foreign_frequency,
            calendar=calendar_foreign,
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

    def _get_payment_date(self, accrual_end: date, calendar: Optional[Calendar]) -> date:
        """Get payment date from accrual end."""
        if calendar:
            adjustment = BusinessDayAdjustment("modified_following", calendar)
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

    def domestic_leg_cash_flows(
        self,
        valuation_date: date,
        projection_curve: Optional[Curve] = None,
    ) -> List[XCCYCashFlow]:
        """Generate domestic leg cash flows."""
        cash_flows = []

        # Initial principal exchange (pay domestic)
        if self.exchange_initial_principal and self.start_date >= valuation_date:
            cash_flows.append(XCCYCashFlow(
                payment_date=self.start_date,
                currency=self.domestic_currency,
                amount=-self.domestic_notional,
                flow_type="principal",
                notional=self.domestic_notional,
            ))

        # Interest payments
        for accrual_start, accrual_end in self._domestic_schedule.periods:
            payment_date = self._get_payment_date(accrual_end, self.calendar_domestic)

            if payment_date < valuation_date:
                continue

            dcf = self.domestic_day_count.year_fraction(accrual_start, accrual_end)

            if self.domestic_is_fixed:
                rate = self.domestic_rate
            else:
                if projection_curve is None:
                    raise ValueError("projection_curve required for floating leg")
                rate = self._project_rate(
                    accrual_start, accrual_end, self._domestic_tenor_years, projection_curve
                ) + self.domestic_spread

            amount = -self.domestic_notional * rate * dcf  # Pay

            cash_flows.append(XCCYCashFlow(
                payment_date=payment_date,
                currency=self.domestic_currency,
                amount=amount,
                flow_type="interest",
                accrual_start=accrual_start,
                accrual_end=accrual_end,
                notional=self.domestic_notional,
                rate=rate,
            ))

        # Final principal exchange (receive domestic)
        if self.exchange_final_principal:
            cash_flows.append(XCCYCashFlow(
                payment_date=self.end_date,
                currency=self.domestic_currency,
                amount=self.domestic_notional,
                flow_type="principal",
                notional=self.domestic_notional,
            ))

        return cash_flows

    def foreign_leg_cash_flows(
        self,
        valuation_date: date,
        projection_curve: Optional[Curve] = None,
    ) -> List[XCCYCashFlow]:
        """Generate foreign leg cash flows."""
        cash_flows = []

        # Initial principal exchange (receive foreign)
        if self.exchange_initial_principal and self.start_date >= valuation_date:
            cash_flows.append(XCCYCashFlow(
                payment_date=self.start_date,
                currency=self.foreign_currency,
                amount=self.foreign_notional,
                flow_type="principal",
                notional=self.foreign_notional,
            ))

        # Interest payments
        for accrual_start, accrual_end in self._foreign_schedule.periods:
            payment_date = self._get_payment_date(accrual_end, self.calendar_foreign)

            if payment_date < valuation_date:
                continue

            dcf = self.foreign_day_count.year_fraction(accrual_start, accrual_end)

            if self.foreign_is_fixed:
                rate = self.foreign_rate
            else:
                if projection_curve is None:
                    raise ValueError("projection_curve required for floating leg")
                rate = self._project_rate(
                    accrual_start, accrual_end, self._foreign_tenor_years, projection_curve
                ) + self.foreign_spread

            amount = self.foreign_notional * rate * dcf  # Receive

            cash_flows.append(XCCYCashFlow(
                payment_date=payment_date,
                currency=self.foreign_currency,
                amount=amount,
                flow_type="interest",
                accrual_start=accrual_start,
                accrual_end=accrual_end,
                notional=self.foreign_notional,
                rate=rate,
            ))

        # Final principal exchange (pay foreign)
        if self.exchange_final_principal:
            cash_flows.append(XCCYCashFlow(
                payment_date=self.end_date,
                currency=self.foreign_currency,
                amount=-self.foreign_notional,
                flow_type="principal",
                notional=self.foreign_notional,
            ))

        return cash_flows

    def present_value(
        self,
        valuation_date: date,
        domestic_discount_curve: Curve,
        foreign_discount_curve: Curve,
        domestic_projection_curve: Optional[Curve] = None,
        foreign_projection_curve: Optional[Curve] = None,
        fx_spot: Optional[float] = None,
    ) -> float:
        """
        Calculate swap NPV in domestic currency.

        Args:
            valuation_date: Valuation date
            domestic_discount_curve: Discount curve for domestic currency
            foreign_discount_curve: Discount curve for foreign currency
            domestic_projection_curve: Projection curve for domestic floating leg
            foreign_projection_curve: Projection curve for foreign floating leg
            fx_spot: Current spot FX rate (foreign per domestic)

        Returns:
            NPV in domestic currency
        """
        if domestic_projection_curve is None:
            domestic_projection_curve = domestic_discount_curve
        if foreign_projection_curve is None:
            foreign_projection_curve = foreign_discount_curve
        if fx_spot is None:
            fx_spot = self.spot_fx

        # Domestic leg PV (in domestic currency)
        domestic_cfs = self.domestic_leg_cash_flows(valuation_date, domestic_projection_curve)
        domestic_pv = sum(cf.discount(domestic_discount_curve) for cf in domestic_cfs)

        # Foreign leg PV (in foreign currency)
        foreign_cfs = self.foreign_leg_cash_flows(valuation_date, foreign_projection_curve)
        foreign_pv = sum(cf.discount(foreign_discount_curve) for cf in foreign_cfs)

        # Convert foreign PV to domestic currency
        foreign_pv_domestic = foreign_pv / fx_spot

        return domestic_pv + foreign_pv_domestic

    def par_spread(
        self,
        valuation_date: date,
        domestic_discount_curve: Curve,
        foreign_discount_curve: Curve,
        domestic_projection_curve: Optional[Curve] = None,
        foreign_projection_curve: Optional[Curve] = None,
        fx_spot: Optional[float] = None,
    ) -> float:
        """
        Calculate par basis spread on foreign leg.

        The spread that makes the swap have zero NPV.
        """
        if domestic_projection_curve is None:
            domestic_projection_curve = domestic_discount_curve
        if foreign_projection_curve is None:
            foreign_projection_curve = foreign_discount_curve
        if fx_spot is None:
            fx_spot = self.spot_fx

        # Save original spread
        original_spread = self.foreign_spread

        # Calculate PV with zero spread
        self.foreign_spread = 0.0
        zero_spread_pv = self.present_value(
            valuation_date,
            domestic_discount_curve,
            foreign_discount_curve,
            domestic_projection_curve,
            foreign_projection_curve,
            fx_spot,
        )

        # Restore spread
        self.foreign_spread = original_spread

        # Calculate foreign leg annuity
        annuity = 0.0
        for accrual_start, accrual_end in self._foreign_schedule.periods:
            payment_date = self._get_payment_date(accrual_end, self.calendar_foreign)

            if payment_date < valuation_date:
                continue

            dcf = self.foreign_day_count.year_fraction(accrual_start, accrual_end)
            df = foreign_discount_curve.discount_factor(payment_date)

            annuity += dcf * df

        annuity_domestic = self.foreign_notional * annuity / fx_spot

        if abs(annuity_domestic) < 1e-10:
            return 0.0

        return -zero_spread_pv / annuity_domestic

    def fx_delta(
        self,
        valuation_date: date,
        domestic_discount_curve: Curve,
        foreign_discount_curve: Curve,
        domestic_projection_curve: Optional[Curve] = None,
        foreign_projection_curve: Optional[Curve] = None,
        fx_spot: Optional[float] = None,
        bump_pct: float = 0.01,
    ) -> float:
        """
        Calculate FX delta (sensitivity to FX rate change).

        Args:
            bump_pct: Percentage bump to FX rate (default 1%)

        Returns:
            Change in NPV for 1% FX move
        """
        if fx_spot is None:
            fx_spot = self.spot_fx

        base_pv = self.present_value(
            valuation_date,
            domestic_discount_curve,
            foreign_discount_curve,
            domestic_projection_curve,
            foreign_projection_curve,
            fx_spot,
        )

        fx_up = fx_spot * (1 + bump_pct)

        bumped_pv = self.present_value(
            valuation_date,
            domestic_discount_curve,
            foreign_discount_curve,
            domestic_projection_curve,
            foreign_projection_curve,
            fx_up,
        )

        return (bumped_pv - base_pv) / bump_pct

    def value(
        self,
        valuation_date: date,
        domestic_discount_curve: Curve,
        foreign_discount_curve: Curve,
        domestic_projection_curve: Optional[Curve] = None,
        foreign_projection_curve: Optional[Curve] = None,
        fx_spot: Optional[float] = None,
    ) -> XCCYSwapResult:
        """Full valuation with all analytics."""
        if domestic_projection_curve is None:
            domestic_projection_curve = domestic_discount_curve
        if foreign_projection_curve is None:
            foreign_projection_curve = foreign_discount_curve
        if fx_spot is None:
            fx_spot = self.spot_fx

        # Cash flows
        domestic_cfs = self.domestic_leg_cash_flows(valuation_date, domestic_projection_curve)
        foreign_cfs = self.foreign_leg_cash_flows(valuation_date, foreign_projection_curve)

        # PVs
        domestic_pv = sum(cf.discount(domestic_discount_curve) for cf in domestic_cfs)
        foreign_pv = sum(cf.discount(foreign_discount_curve) for cf in foreign_cfs)
        foreign_pv_domestic = foreign_pv / fx_spot

        total_pv = domestic_pv + foreign_pv_domestic

        # FX delta
        fx_delta = self.fx_delta(
            valuation_date,
            domestic_discount_curve,
            foreign_discount_curve,
            domestic_projection_curve,
            foreign_projection_curve,
            fx_spot,
        )

        # IR deltas (1bp parallel shift)
        base_pv = total_pv

        # Domestic IR delta
        dom_disc_up = domestic_discount_curve.shift(1.0)
        dom_proj_up = domestic_projection_curve.shift(1.0)
        pv_dom_up = self.present_value(
            valuation_date, dom_disc_up, foreign_discount_curve,
            dom_proj_up, foreign_projection_curve, fx_spot
        )
        ir_delta_domestic = pv_dom_up - base_pv

        # Foreign IR delta
        for_disc_up = foreign_discount_curve.shift(1.0)
        for_proj_up = foreign_projection_curve.shift(1.0)
        pv_for_up = self.present_value(
            valuation_date, domestic_discount_curve, for_disc_up,
            domestic_projection_curve, for_proj_up, fx_spot
        )
        ir_delta_foreign = pv_for_up - base_pv

        return XCCYSwapResult(
            npv_domestic=domestic_pv,
            npv_foreign=foreign_pv,
            npv_total=total_pv,
            domestic_leg_pv=domestic_pv,
            foreign_leg_pv=foreign_pv_domestic,
            fx_delta=fx_delta,
            ir_delta_domestic=ir_delta_domestic,
            ir_delta_foreign=ir_delta_foreign,
            cash_flows_domestic=domestic_cfs,
            cash_flows_foreign=foreign_cfs,
        )

    def __repr__(self) -> str:
        return (
            f"CrossCurrencySwap("
            f"{self.domestic_currency}/{self.foreign_currency}, "
            f"notional={self.domestic_notional:,.0f} {self.domestic_currency}, "
            f"fx={self.spot_fx:.4f}, "
            f"{self.start_date} to {self.end_date})"
        )


class MTMCrossCurrencySwap(CrossCurrencySwap):
    """
    Mark-to-Market Cross-Currency Swap.

    In an MTM XCCY swap, the foreign notional is reset at each payment date
    to reflect the current spot FX rate. This reduces FX risk by:

    1. Resetting foreign notional: foreign_notional_new = domestic_notional * current_fx
    2. Paying/receiving the difference as MTM adjustment

    This is the standard structure for interbank XCCY swaps.
    """

    def __init__(
        self,
        domestic_notional: float,
        domestic_currency: str,
        foreign_currency: str,
        start_date: date,
        end_date: date,
        spot_fx: float,
        reset_frequency: Union[str, Frequency] = Frequency.QUARTERLY,
        **kwargs,
    ):
        """
        Initialize MTM cross-currency swap.

        Args:
            domestic_notional: Notional in domestic currency
            domestic_currency: Domestic currency code
            foreign_currency: Foreign currency code
            start_date: Effective date
            end_date: Maturity date
            spot_fx: Initial spot FX rate
            reset_frequency: Frequency of MTM resets
            **kwargs: Other arguments passed to CrossCurrencySwap
        """
        super().__init__(
            domestic_notional=domestic_notional,
            domestic_currency=domestic_currency,
            foreign_currency=foreign_currency,
            start_date=start_date,
            end_date=end_date,
            spot_fx=spot_fx,
            **kwargs,
        )

        # Parse reset frequency
        if isinstance(reset_frequency, str):
            reset_frequency = Frequency[reset_frequency.upper()]

        self.reset_frequency = reset_frequency

        # Generate reset schedule
        self._reset_schedule = Schedule(
            effective_date=start_date,
            termination_date=end_date,
            frequency=reset_frequency,
            calendar=self.calendar_foreign,
        )

    def _get_reset_dates(self) -> List[date]:
        """Get all reset dates (accrual end dates)."""
        return [end for _, end in self._reset_schedule.periods]

    def _calculate_mtm_adjustments(
        self,
        valuation_date: date,
        fx_forward_curve: Optional[Dict[date, float]] = None,
        current_fx: Optional[float] = None,
    ) -> List[XCCYCashFlow]:
        """
        Calculate MTM adjustment cash flows.

        At each reset date:
        - New notional = domestic_notional * fx_reset
        - MTM adjustment = new_notional - old_notional
        """
        if current_fx is None:
            current_fx = self.spot_fx

        fx_forward_curve = fx_forward_curve or {}

        adjustments = []
        prev_notional = self.foreign_notional

        for _, reset_date in self._reset_schedule.periods[:-1]:  # Exclude final
            if reset_date < valuation_date:
                # Past reset - use historical or estimate
                fx_rate = fx_forward_curve.get(reset_date, self.spot_fx)
            elif reset_date == valuation_date:
                fx_rate = current_fx
            else:
                # Future reset - use forward FX
                fx_rate = fx_forward_curve.get(reset_date, current_fx)

            new_notional = self.domestic_notional * fx_rate
            adjustment = new_notional - prev_notional

            if abs(adjustment) > 1e-6:
                adjustments.append(XCCYCashFlow(
                    payment_date=reset_date,
                    currency=self.foreign_currency,
                    amount=adjustment,  # Receive if positive (FX moved in our favor)
                    flow_type="mtm_adjustment",
                    notional=new_notional,
                ))

            prev_notional = new_notional

        return adjustments

    def foreign_leg_cash_flows(
        self,
        valuation_date: date,
        projection_curve: Optional[Curve] = None,
        fx_forward_curve: Optional[Dict[date, float]] = None,
    ) -> List[XCCYCashFlow]:
        """
        Generate foreign leg cash flows including MTM adjustments.

        Args:
            valuation_date: Valuation date
            projection_curve: Curve for floating rate projection
            fx_forward_curve: Dict of {date: fx_rate} for forward FX rates
        """
        # Base cash flows
        cash_flows = super().foreign_leg_cash_flows(valuation_date, projection_curve)

        # Add MTM adjustments
        mtm_adjustments = self._calculate_mtm_adjustments(
            valuation_date, fx_forward_curve
        )
        cash_flows.extend(mtm_adjustments)

        # Sort by date
        cash_flows.sort(key=lambda cf: cf.payment_date)

        return cash_flows

    def present_value(
        self,
        valuation_date: date,
        domestic_discount_curve: Curve,
        foreign_discount_curve: Curve,
        domestic_projection_curve: Optional[Curve] = None,
        foreign_projection_curve: Optional[Curve] = None,
        fx_spot: Optional[float] = None,
        fx_forward_curve: Optional[Dict[date, float]] = None,
    ) -> float:
        """
        Calculate MTM swap NPV in domestic currency.

        Args:
            valuation_date: Valuation date
            domestic_discount_curve: Discount curve for domestic currency
            foreign_discount_curve: Discount curve for foreign currency
            domestic_projection_curve: Projection curve for domestic leg
            foreign_projection_curve: Projection curve for foreign leg
            fx_spot: Current spot FX rate
            fx_forward_curve: Forward FX rates for MTM resets
        """
        if domestic_projection_curve is None:
            domestic_projection_curve = domestic_discount_curve
        if foreign_projection_curve is None:
            foreign_projection_curve = foreign_discount_curve
        if fx_spot is None:
            fx_spot = self.spot_fx

        # Build FX forward curve from interest rate differential if not provided
        if fx_forward_curve is None:
            fx_forward_curve = self._build_fx_forward_curve(
                valuation_date,
                domestic_discount_curve,
                foreign_discount_curve,
                fx_spot
            )

        # Domestic leg PV
        domestic_cfs = self.domestic_leg_cash_flows(valuation_date, domestic_projection_curve)
        domestic_pv = sum(cf.discount(domestic_discount_curve) for cf in domestic_cfs)

        # Foreign leg PV including MTM adjustments
        foreign_cfs = self.foreign_leg_cash_flows(
            valuation_date, foreign_projection_curve, fx_forward_curve
        )
        foreign_pv = sum(cf.discount(foreign_discount_curve) for cf in foreign_cfs)

        # Convert to domestic currency
        foreign_pv_domestic = foreign_pv / fx_spot

        return domestic_pv + foreign_pv_domestic

    def _build_fx_forward_curve(
        self,
        valuation_date: date,
        domestic_discount_curve: Curve,
        foreign_discount_curve: Curve,
        fx_spot: float,
    ) -> Dict[date, float]:
        """
        Build FX forward curve from interest rate curves using Covered Interest Parity.

        The FX quote convention is: foreign currency per domestic currency
        (e.g., if domestic=USD, foreign=EUR, spot_fx=0.91 means 0.91 EUR per USD)

        Covered Interest Rate Parity (CIP):
        F(t,T) / S(t) = DF_foreign(t,T) / DF_domestic(t,T)

        Therefore:
        Forward FX = Spot × DF_foreign / DF_domestic

        If domestic rates > foreign rates:
            DF_dom < DF_for → Forward > Spot (domestic appreciates)
        If foreign rates > domestic rates:
            DF_for < DF_dom → Forward < Spot (domestic depreciates)
        """
        fx_forwards = {}

        for _, reset_date in self._reset_schedule.periods:
            if reset_date <= valuation_date:
                continue

            t = (reset_date - valuation_date).days / 365.0

            df_dom = domestic_discount_curve.discount_factor(t)
            df_for = foreign_discount_curve.discount_factor(t)

            # FX forward using CIP: F = S × DF_for / DF_dom
            fx_fwd = fx_spot * df_for / df_dom
            fx_forwards[reset_date] = fx_fwd

        return fx_forwards

    def __repr__(self) -> str:
        return (
            f"MTMCrossCurrencySwap("
            f"{self.domestic_currency}/{self.foreign_currency}, "
            f"notional={self.domestic_notional:,.0f} {self.domestic_currency}, "
            f"reset_freq={self.reset_frequency.name}, "
            f"{self.start_date} to {self.end_date})"
        )


class CrossCurrencyBasisSwap:
    """
    Cross-Currency Basis Swap.

    Exchange of floating rates in two currencies with a basis spread
    on one leg. Both legs are floating (typically overnight or 3M).

    This is used to:
    - Trade the basis spread between currency curves
    - Hedge cross-currency funding costs
    - Speculate on currency basis movements

    Example:
        - Pay USD SOFR flat
        - Receive EUR EURIBOR 3M + basis spread
    """

    def __init__(
        self,
        domestic_notional: float,
        domestic_currency: str,
        foreign_currency: str,
        start_date: date,
        end_date: date,
        spot_fx: float,
        basis_spread: float = 0.0,  # Spread on foreign leg
        spread_on_domestic: bool = False,
        # Domestic leg settings
        domestic_index_tenor: str = "ON",  # Overnight for RFR
        domestic_frequency: Union[str, Frequency] = Frequency.QUARTERLY,
        domestic_day_count: Union[str, DayCountConvention] = "ACT/360",
        # Foreign leg settings
        foreign_index_tenor: str = "3M",
        foreign_frequency: Union[str, Frequency] = Frequency.QUARTERLY,
        foreign_day_count: Union[str, DayCountConvention] = "ACT/360",
        # Principal exchange
        exchange_initial_principal: bool = True,
        exchange_final_principal: bool = True,
        # Calendars
        calendar_domestic: Optional[Calendar] = None,
        calendar_foreign: Optional[Calendar] = None,
    ):
        """
        Initialize cross-currency basis swap.

        Args:
            domestic_notional: Notional in domestic currency
            domestic_currency: Domestic currency code
            foreign_currency: Foreign currency code
            start_date: Effective date
            end_date: Maturity date
            spot_fx: Spot FX rate (foreign per domestic)
            basis_spread: Basis spread (on foreign leg by default)
            spread_on_domestic: If True, spread is on domestic leg
            domestic_index_tenor: Domestic floating rate tenor
            domestic_frequency: Domestic payment frequency
            domestic_day_count: Domestic day count
            foreign_index_tenor: Foreign floating rate tenor
            foreign_frequency: Foreign payment frequency
            foreign_day_count: Foreign day count
            exchange_initial_principal: Exchange notional at start
            exchange_final_principal: Exchange notional at maturity
            calendar_domestic: Calendar for domestic leg
            calendar_foreign: Calendar for foreign leg
        """
        self.domestic_notional = domestic_notional
        self.domestic_currency = domestic_currency.upper()
        self.foreign_currency = foreign_currency.upper()
        self.start_date = start_date
        self.end_date = end_date
        self.spot_fx = spot_fx
        self.basis_spread = basis_spread
        self.spread_on_domestic = spread_on_domestic

        # Calculate foreign notional
        self.foreign_notional = domestic_notional * spot_fx

        self.domestic_index_tenor = domestic_index_tenor
        self.foreign_index_tenor = foreign_index_tenor

        self.exchange_initial_principal = exchange_initial_principal
        self.exchange_final_principal = exchange_final_principal

        # Day counts
        if isinstance(domestic_day_count, str):
            domestic_day_count = _get_convention(domestic_day_count)
        if isinstance(foreign_day_count, str):
            foreign_day_count = _get_convention(foreign_day_count)

        self.domestic_day_count = domestic_day_count
        self.foreign_day_count = foreign_day_count

        # Parse frequencies
        if isinstance(domestic_frequency, str):
            domestic_frequency = Frequency[domestic_frequency.upper()]
        if isinstance(foreign_frequency, str):
            foreign_frequency = Frequency[foreign_frequency.upper()]

        self.domestic_frequency = domestic_frequency
        self.foreign_frequency = foreign_frequency

        self.calendar_domestic = calendar_domestic
        self.calendar_foreign = calendar_foreign

        # Parse index tenors
        self._domestic_tenor_years = self._parse_tenor(domestic_index_tenor)
        self._foreign_tenor_years = self._parse_tenor(foreign_index_tenor)

        # Generate schedules
        self._domestic_schedule = Schedule(
            effective_date=start_date,
            termination_date=end_date,
            frequency=domestic_frequency,
            calendar=calendar_domestic,
        )

        self._foreign_schedule = Schedule(
            effective_date=start_date,
            termination_date=end_date,
            frequency=foreign_frequency,
            calendar=calendar_foreign,
        )

    @staticmethod
    def _parse_tenor(tenor: str) -> float:
        """Parse tenor string to year fraction."""
        tenor = tenor.upper().strip()
        if tenor == "ON" or tenor == "O/N":
            return 1/365  # Overnight
        elif tenor.endswith("M"):
            return int(tenor[:-1]) / 12.0
        elif tenor.endswith("Y"):
            return float(tenor[:-1])
        elif tenor.endswith("W"):
            return int(tenor[:-1]) / 52.0
        elif tenor.endswith("D"):
            return int(tenor[:-1]) / 365.0
        return 0.25

    def _get_payment_date(self, accrual_end: date, calendar: Optional[Calendar]) -> date:
        """Get payment date from accrual end."""
        if calendar:
            adjustment = BusinessDayAdjustment("modified_following", calendar)
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

    def domestic_leg_pv(
        self,
        valuation_date: date,
        projection_curve: Curve,
        discount_curve: Curve,
    ) -> float:
        """Calculate domestic floating leg PV."""
        pv = 0.0
        spread = self.basis_spread if self.spread_on_domestic else 0.0

        # Initial principal (pay)
        if self.exchange_initial_principal and self.start_date >= valuation_date:
            df = discount_curve.discount_factor(self.start_date)
            pv -= self.domestic_notional * df

        # Interest payments
        for accrual_start, accrual_end in self._domestic_schedule.periods:
            payment_date = self._get_payment_date(accrual_end, self.calendar_domestic)

            if payment_date < valuation_date:
                continue

            rate = self._project_rate(
                accrual_start, accrual_end, self._domestic_tenor_years, projection_curve
            )

            dcf = self.domestic_day_count.year_fraction(accrual_start, accrual_end)
            df = discount_curve.discount_factor(payment_date)

            pv -= self.domestic_notional * (rate + spread) * dcf * df

        # Final principal (receive)
        if self.exchange_final_principal:
            df = discount_curve.discount_factor(self.end_date)
            pv += self.domestic_notional * df

        return pv

    def foreign_leg_pv(
        self,
        valuation_date: date,
        projection_curve: Curve,
        discount_curve: Curve,
    ) -> float:
        """Calculate foreign floating leg PV."""
        pv = 0.0
        spread = self.basis_spread if not self.spread_on_domestic else 0.0

        # Initial principal (receive)
        if self.exchange_initial_principal and self.start_date >= valuation_date:
            df = discount_curve.discount_factor(self.start_date)
            pv += self.foreign_notional * df

        # Interest payments
        for accrual_start, accrual_end in self._foreign_schedule.periods:
            payment_date = self._get_payment_date(accrual_end, self.calendar_foreign)

            if payment_date < valuation_date:
                continue

            rate = self._project_rate(
                accrual_start, accrual_end, self._foreign_tenor_years, projection_curve
            )

            dcf = self.foreign_day_count.year_fraction(accrual_start, accrual_end)
            df = discount_curve.discount_factor(payment_date)

            pv += self.foreign_notional * (rate + spread) * dcf * df

        # Final principal (pay)
        if self.exchange_final_principal:
            df = discount_curve.discount_factor(self.end_date)
            pv -= self.foreign_notional * df

        return pv

    def present_value(
        self,
        valuation_date: date,
        domestic_projection_curve: Curve,
        domestic_discount_curve: Curve,
        foreign_projection_curve: Curve,
        foreign_discount_curve: Curve,
        fx_spot: Optional[float] = None,
    ) -> float:
        """
        Calculate swap NPV in domestic currency.

        Args:
            valuation_date: Valuation date
            domestic_projection_curve: Projection curve for domestic floating leg
            domestic_discount_curve: Discount curve for domestic currency
            foreign_projection_curve: Projection curve for foreign floating leg
            foreign_discount_curve: Discount curve for foreign currency
            fx_spot: Current spot FX rate

        Returns:
            NPV in domestic currency
        """
        if fx_spot is None:
            fx_spot = self.spot_fx

        domestic_pv = self.domestic_leg_pv(
            valuation_date, domestic_projection_curve, domestic_discount_curve
        )

        foreign_pv = self.foreign_leg_pv(
            valuation_date, foreign_projection_curve, foreign_discount_curve
        )

        # Convert foreign PV to domestic
        foreign_pv_domestic = foreign_pv / fx_spot

        return domestic_pv + foreign_pv_domestic

    def par_spread(
        self,
        valuation_date: date,
        domestic_projection_curve: Curve,
        domestic_discount_curve: Curve,
        foreign_projection_curve: Curve,
        foreign_discount_curve: Curve,
        fx_spot: Optional[float] = None,
    ) -> float:
        """Calculate par basis spread."""
        if fx_spot is None:
            fx_spot = self.spot_fx

        # Save original spread
        original_spread = self.basis_spread

        # Calculate PV with zero spread
        self.basis_spread = 0.0
        zero_spread_pv = self.present_value(
            valuation_date,
            domestic_projection_curve,
            domestic_discount_curve,
            foreign_projection_curve,
            foreign_discount_curve,
            fx_spot,
        )

        # Restore spread
        self.basis_spread = original_spread

        # Calculate annuity for spread leg
        if self.spread_on_domestic:
            annuity = 0.0
            for accrual_start, accrual_end in self._domestic_schedule.periods:
                payment_date = self._get_payment_date(accrual_end, self.calendar_domestic)
                if payment_date < valuation_date:
                    continue
                dcf = self.domestic_day_count.year_fraction(accrual_start, accrual_end)
                df = domestic_discount_curve.discount_factor(payment_date)
                annuity += dcf * df
            annuity *= self.domestic_notional
        else:
            annuity = 0.0
            for accrual_start, accrual_end in self._foreign_schedule.periods:
                payment_date = self._get_payment_date(accrual_end, self.calendar_foreign)
                if payment_date < valuation_date:
                    continue
                dcf = self.foreign_day_count.year_fraction(accrual_start, accrual_end)
                df = foreign_discount_curve.discount_factor(payment_date)
                annuity += dcf * df
            annuity *= self.foreign_notional / fx_spot

        if abs(annuity) < 1e-10:
            return 0.0

        return -zero_spread_pv / annuity

    def __repr__(self) -> str:
        spread_leg = self.domestic_currency if self.spread_on_domestic else self.foreign_currency
        return (
            f"CrossCurrencyBasisSwap("
            f"{self.domestic_currency}/{self.foreign_currency}, "
            f"notional={self.domestic_notional:,.0f} {self.domestic_currency}, "
            f"spread={self.basis_spread*10000:.1f}bp on {spread_leg}, "
            f"{self.start_date} to {self.end_date})"
        )
