"""
Credit Default Swaps (CDS)

Implements:
- Single-name CDS pricing
- CDS spread calculation
- Protection and premium leg valuation
- CDS curve bootstrapping
- Risk metrics (spread DV01, jump-to-default)

A CDS is a contract where:
- Protection buyer pays periodic premium (spread) to protection seller
- If reference entity defaults, protection seller pays (1 - R) * Notional
- R is recovery rate
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Union, Optional, List
from enum import Enum
from scipy.optimize import brentq

from finceptquantlib.curves.curve import Curve
from finceptquantlib.scheduling.schedule import Schedule, Frequency
from finceptquantlib.scheduling.calendar import Calendar
from finceptquantlib.scheduling.daycount import DayCountConvention, _get_convention
from .hazard import (
    SurvivalCurve,
    PiecewiseHazardRate,
    ConstantHazardRate,
)


class CDSType(Enum):
    """CDS position type."""
    PROTECTION_BUYER = "protection_buyer"   # Pays spread, receives protection
    PROTECTION_SELLER = "protection_seller" # Receives spread, pays protection


@dataclass
class CDSResult:
    """Result of CDS valuation."""
    present_value: float
    clean_price: float          # 100 - upfront_percent
    upfront_percent: float      # Upfront payment as % of notional
    spread_dv01: float          # Sensitivity to 1bp spread change
    ir_dv01: float              # Sensitivity to 1bp rate change
    premium_leg_pv: float
    protection_leg_pv: float
    accrued_premium: float
    risky_annuity: float        # RPV01
    par_spread: float           # Break-even spread


class CreditDefaultSwap:
    """
    Credit Default Swap.

    A CDS has two legs:
    1. Premium Leg: Periodic spread payments contingent on survival
    2. Protection Leg: Payment of (1-R)*N upon default

    Standard CDS conventions:
    - Premium paid quarterly, ACT/360
    - Standard coupon dates (IMM dates: Mar 20, Jun 20, Sep 20, Dec 20)
    - Standard coupons: 100bp or 500bp
    - Upfront payment settles the difference from par
    """

    def __init__(
        self,
        notional: float,
        effective_date: date,
        maturity_date: date,
        spread: float,
        cds_type: Union[str, CDSType] = CDSType.PROTECTION_BUYER,
        recovery_rate: float = 0.4,
        frequency: Union[str, Frequency] = Frequency.QUARTERLY,
        day_count: Union[str, DayCountConvention] = "ACT/360",
        calendar: Optional[Calendar] = None,
        pay_at_default: bool = True,
        accrual_on_default: bool = True,
    ):
        """
        Initialize CDS.

        Args:
            notional: Notional principal
            effective_date: Protection start date
            maturity_date: Protection end date
            spread: Running spread (annualized, in decimal, e.g., 0.01 for 100bp)
            cds_type: PROTECTION_BUYER or PROTECTION_SELLER
            recovery_rate: Assumed recovery rate
            frequency: Premium payment frequency
            day_count: Day count convention
            calendar: Business day calendar
            pay_at_default: If True, protection paid at default time
            accrual_on_default: If True, accrued premium paid on default
        """
        self.notional = notional
        self.effective_date = effective_date
        self.maturity_date = maturity_date
        self.spread = spread
        self.recovery_rate = recovery_rate

        if isinstance(cds_type, str):
            cds_type = CDSType[cds_type.upper()]
        self.cds_type = cds_type

        if isinstance(frequency, str):
            frequency = Frequency[frequency.upper()]
        self.frequency = frequency

        if isinstance(day_count, str):
            day_count = _get_convention(day_count)
        self.day_count = day_count

        self.calendar = calendar
        self.pay_at_default = pay_at_default
        self.accrual_on_default = accrual_on_default

        # Generate premium schedule
        self._schedule = Schedule(
            effective_date=effective_date,
            termination_date=maturity_date,
            frequency=frequency,
            calendar=calendar,
        )

    @property
    def loss_given_default(self) -> float:
        """Loss given default = 1 - Recovery Rate."""
        return 1.0 - self.recovery_rate

    def premium_leg_pv(
        self,
        valuation_date: date,
        discount_curve: Curve,
        survival_curve: SurvivalCurve,
    ) -> float:
        """
        Calculate present value of premium leg.

        PV = Σ S(tᵢ) * DF(tᵢ) * Δtᵢ * Spread * Notional
             + accrual on default (if applicable)
        """
        total = 0.0

        for period in self._schedule.periods:
            payment_date = period.accrual_end

            if payment_date <= valuation_date:
                continue

            # Day count fraction
            dcf = self.day_count.year_fraction(period.accrual_start, period.accrual_end)

            # Discount factor and survival probability
            df = discount_curve.discount_factor(payment_date)
            sp = survival_curve.survival_probability(period.accrual_end)

            # Premium payment if survived
            total += self.spread * dcf * df * sp * self.notional

            # Accrual on default (approximate)
            if self.accrual_on_default:
                # Expected accrual payment at midpoint of period
                mid_date = period.accrual_start + timedelta(days=(period.accrual_end - period.accrual_start).days // 2)
                mid_dcf = self.day_count.year_fraction(period.accrual_start, mid_date)

                sp_start = survival_curve.survival_probability(period.accrual_start)
                sp_end = survival_curve.survival_probability(period.accrual_end)
                default_prob = sp_start - sp_end

                df_mid = discount_curve.discount_factor(mid_date)

                total += self.spread * mid_dcf * df_mid * default_prob * self.notional

        return total

    def protection_leg_pv(
        self,
        valuation_date: date,
        discount_curve: Curve,
        survival_curve: SurvivalCurve,
    ) -> float:
        """
        Calculate present value of protection leg.

        PV = LGD * Notional * Σ (S(tᵢ₋₁) - S(tᵢ)) * DF(midᵢ)

        where LGD = 1 - Recovery Rate
        """
        total = 0.0
        lgd = self.loss_given_default

        for period in self._schedule.periods:
            if period.accrual_end <= valuation_date:
                continue

            # Default probability in this period
            sp_start = survival_curve.survival_probability(
                max(period.accrual_start, valuation_date)
            )
            sp_end = survival_curve.survival_probability(period.accrual_end)
            default_prob = max(sp_start - sp_end, 0)

            if default_prob <= 0:
                continue

            # Payment at midpoint of period (approximation)
            if self.pay_at_default:
                mid_date = period.accrual_start + timedelta(
                    days=(period.accrual_end - period.accrual_start).days // 2
                )
            else:
                mid_date = period.accrual_end

            df = discount_curve.discount_factor(mid_date)

            total += lgd * self.notional * default_prob * df

        return total

    def present_value(
        self,
        valuation_date: date,
        discount_curve: Curve,
        survival_curve: SurvivalCurve,
    ) -> float:
        """
        Calculate CDS present value.

        For protection buyer: PV = Protection Leg - Premium Leg
        For protection seller: PV = Premium Leg - Protection Leg
        """
        premium_pv = self.premium_leg_pv(valuation_date, discount_curve, survival_curve)
        protection_pv = self.protection_leg_pv(valuation_date, discount_curve, survival_curve)

        if self.cds_type == CDSType.PROTECTION_BUYER:
            return protection_pv - premium_pv
        else:
            return premium_pv - protection_pv

    def risky_annuity(
        self,
        valuation_date: date,
        discount_curve: Curve,
        survival_curve: SurvivalCurve,
    ) -> float:
        """
        Calculate risky annuity (RPV01).

        The present value of receiving 1bp on the premium leg.
        Also known as risky PV01 or duration.
        """
        total = 0.0

        for period in self._schedule.periods:
            if period.accrual_end <= valuation_date:
                continue

            dcf = self.day_count.year_fraction(period.accrual_start, period.accrual_end)
            df = discount_curve.discount_factor(period.accrual_end)
            sp = survival_curve.survival_probability(period.accrual_end)

            total += dcf * df * sp

            # Accrual on default component
            if self.accrual_on_default:
                mid_date = period.accrual_start + timedelta(
                    days=(period.accrual_end - period.accrual_start).days // 2
                )
                mid_dcf = self.day_count.year_fraction(period.accrual_start, mid_date)
                sp_start = survival_curve.survival_probability(period.accrual_start)
                default_prob = sp_start - sp
                df_mid = discount_curve.discount_factor(mid_date)
                total += mid_dcf * df_mid * default_prob

        return total * self.notional

    def par_spread(
        self,
        valuation_date: date,
        discount_curve: Curve,
        survival_curve: SurvivalCurve,
    ) -> float:
        """
        Calculate par (break-even) spread.

        The spread at which the CDS has zero present value.
        Par Spread = Protection Leg PV / Risky Annuity
        """
        protection_pv = self.protection_leg_pv(
            valuation_date, discount_curve, survival_curve
        )
        annuity = self.risky_annuity(valuation_date, discount_curve, survival_curve)

        if abs(annuity) < 1e-10:
            return 0.0

        return protection_pv / annuity

    def upfront_payment(
        self,
        valuation_date: date,
        discount_curve: Curve,
        survival_curve: SurvivalCurve,
    ) -> float:
        """
        Calculate upfront payment.

        Upfront = (Par Spread - Contract Spread) * Risky Annuity

        Positive upfront means protection buyer pays upfront.
        """
        par = self.par_spread(valuation_date, discount_curve, survival_curve)
        annuity = self.risky_annuity(valuation_date, discount_curve, survival_curve)

        return (par - self.spread) * annuity

    def clean_price(
        self,
        valuation_date: date,
        discount_curve: Curve,
        survival_curve: SurvivalCurve,
    ) -> float:
        """
        Calculate clean price (market convention).

        Clean Price = 100 - Upfront %

        where Upfront % is upfront payment as % of notional.
        """
        upfront = self.upfront_payment(valuation_date, discount_curve, survival_curve)
        upfront_pct = upfront / self.notional * 100
        return 100 - upfront_pct

    def accrued_premium(self, valuation_date: date) -> float:
        """Calculate accrued premium on valuation date."""
        # Find current period
        for period in self._schedule.periods:
            if period.accrual_start <= valuation_date < period.accrual_end:
                dcf = self.day_count.year_fraction(period.accrual_start, valuation_date)
                return self.spread * dcf * self.notional

        return 0.0

    def spread_dv01(
        self,
        valuation_date: date,
        discount_curve: Curve,
        survival_curve: SurvivalCurve,
        bump_size: float = 1.0,
    ) -> float:
        """
        Calculate spread DV01.

        Sensitivity to 1bp parallel shift in credit spread.
        """
        # Bump hazard rates (which affects survival curve)
        if isinstance(survival_curve, (PiecewiseHazardRate, ConstantHazardRate)):
            bumped_curve = survival_curve.shift(bump_size)
        else:
            # For other survival curves, approximate by bumping spread
            return self.risky_annuity(valuation_date, discount_curve, survival_curve) / 10000

        base_pv = self.present_value(valuation_date, discount_curve, survival_curve)
        bumped_pv = self.present_value(valuation_date, discount_curve, bumped_curve)

        return bumped_pv - base_pv

    def ir_dv01(
        self,
        valuation_date: date,
        discount_curve: Curve,
        survival_curve: SurvivalCurve,
        bump_size: float = 1.0,
    ) -> float:
        """
        Calculate interest rate DV01.

        Sensitivity to 1bp parallel shift in interest rates.
        """
        base_pv = self.present_value(valuation_date, discount_curve, survival_curve)

        bumped_discount = discount_curve.shift(bump_size)
        bumped_pv = self.present_value(valuation_date, bumped_discount, survival_curve)

        return bumped_pv - base_pv

    def jump_to_default(
        self,
        valuation_date: date,
        discount_curve: Curve,
    ) -> float:
        """
        Calculate jump-to-default (JTD) risk.

        The loss if default occurs immediately.

        For protection buyer: JTD = (1 - R) * N - Accrued
        For protection seller: JTD = -[(1 - R) * N - Accrued]
        """
        protection_payout = self.loss_given_default * self.notional
        accrued = self.accrued_premium(valuation_date)

        if self.cds_type == CDSType.PROTECTION_BUYER:
            return protection_payout - accrued
        else:
            return -(protection_payout - accrued)

    def value(
        self,
        valuation_date: date,
        discount_curve: Curve,
        survival_curve: SurvivalCurve,
    ) -> CDSResult:
        """
        Full valuation with all analytics.

        Returns:
            CDSResult with all metrics
        """
        pv = self.present_value(valuation_date, discount_curve, survival_curve)
        premium_pv = self.premium_leg_pv(valuation_date, discount_curve, survival_curve)
        protection_pv = self.protection_leg_pv(valuation_date, discount_curve, survival_curve)
        par = self.par_spread(valuation_date, discount_curve, survival_curve)
        annuity = self.risky_annuity(valuation_date, discount_curve, survival_curve)
        upfront = self.upfront_payment(valuation_date, discount_curve, survival_curve)
        accrued = self.accrued_premium(valuation_date)
        spread_dv01 = self.spread_dv01(valuation_date, discount_curve, survival_curve)
        ir_dv01 = self.ir_dv01(valuation_date, discount_curve, survival_curve)

        upfront_pct = upfront / self.notional * 100
        clean = 100 - upfront_pct

        return CDSResult(
            present_value=pv,
            clean_price=clean,
            upfront_percent=upfront_pct,
            spread_dv01=spread_dv01,
            ir_dv01=ir_dv01,
            premium_leg_pv=premium_pv,
            protection_leg_pv=protection_pv,
            accrued_premium=accrued,
            risky_annuity=annuity,
            par_spread=par,
        )

    def __repr__(self) -> str:
        return (f"CDS({self.cds_type.value}, "
                f"notional={self.notional:,.0f}, "
                f"spread={self.spread * 10000:.0f}bp, "
                f"{self.effective_date} to {self.maturity_date})")


class CDSBootstrap:
    """
    Bootstrap hazard rates from CDS spreads.

    Given a set of CDS spreads at different tenors,
    calibrate a piecewise hazard rate curve.
    """

    def __init__(
        self,
        reference_date: date,
        discount_curve: Curve,
        recovery_rate: float = 0.4,
        day_count: Union[str, DayCountConvention] = "ACT/360",
    ):
        """
        Initialize bootstrapper.

        Args:
            reference_date: Valuation date
            discount_curve: Risk-free discount curve
            recovery_rate: Recovery rate assumption
            day_count: Day count convention
        """
        self.reference_date = reference_date
        self.discount_curve = discount_curve
        self.recovery_rate = recovery_rate

        if isinstance(day_count, str):
            day_count = _get_convention(day_count)
        self.day_count = day_count

    def bootstrap(
        self,
        tenors: List[float],
        spreads: List[float],
    ) -> PiecewiseHazardRate:
        """
        Bootstrap hazard curve from CDS spreads.

        Args:
            tenors: CDS tenors in years (e.g., [1, 2, 3, 5, 7, 10])
            spreads: Par CDS spreads (in decimal, e.g., 0.01 for 100bp)

        Returns:
            Calibrated PiecewiseHazardRate curve
        """
        # Sort by tenor
        combined = sorted(zip(tenors, spreads), key=lambda x: x[0])
        tenors = [t for t, s in combined]
        spreads = [s for t, s in combined]

        hazard_rates = []

        for i, (tenor, spread) in enumerate(zip(tenors, spreads)):
            # Maturity date for this CDS
            maturity = self.reference_date + timedelta(days=int(tenor * 365))

            # Use already-calibrated hazard rates for earlier tenors
            if i == 0:
                prev_hazard_rates = []
                prev_tenors = []
            else:
                prev_hazard_rates = hazard_rates.copy()
                prev_tenors = tenors[:i]

            # Find hazard rate that makes this CDS trade at par
            def objective(
                h: float,
                prev_tenors: List[float] = prev_tenors,
                prev_hazard_rates: List[float] = prev_hazard_rates,
                tenor: float = tenor,
                maturity: date = maturity,
                spread: float = spread,
            ) -> float:
                # Build survival curve with new hazard rate
                test_tenors = prev_tenors + [tenor]
                test_hazards = prev_hazard_rates + [h]

                survival = PiecewiseHazardRate(
                    reference_date=self.reference_date,
                    tenors=test_tenors,
                    hazard_rates=test_hazards,
                    day_count=self.day_count,
                )

                # Create CDS and get par spread
                cds = CreditDefaultSwap(
                    notional=10_000_000,
                    effective_date=self.reference_date,
                    maturity_date=maturity,
                    spread=spread,
                    recovery_rate=self.recovery_rate,
                    day_count=self.day_count,
                )

                par = cds.par_spread(
                    self.reference_date,
                    self.discount_curve,
                    survival,
                )

                return par - spread

            # Initial guess based on spread / (1 - R)
            h_guess = spread / (1 - self.recovery_rate)

            try:
                h_calibrated = brentq(objective, 0.0001, 1.0)
            except ValueError:
                # Fall back to initial guess
                h_calibrated = h_guess

            hazard_rates.append(h_calibrated)

        return PiecewiseHazardRate(
            reference_date=self.reference_date,
            tenors=tenors,
            hazard_rates=hazard_rates,
            day_count=self.day_count,
        )


class CDSIndex:
    """
    CDS Index (e.g., CDX, iTraxx).

    A portfolio of single-name CDS with standardized terms.
    """

    def __init__(
        self,
        notional: float,
        effective_date: date,
        maturity_date: date,
        spread: float,
        num_constituents: int = 125,
        recovery_rate: float = 0.4,
        day_count: Union[str, DayCountConvention] = "ACT/360",
    ):
        """
        Initialize CDS index.

        Args:
            notional: Total notional
            effective_date: Index effective date
            maturity_date: Index maturity date
            spread: Running spread
            num_constituents: Number of reference entities
            recovery_rate: Assumed recovery rate
            day_count: Day count convention
        """
        self.notional = notional
        self.effective_date = effective_date
        self.maturity_date = maturity_date
        self.spread = spread
        self.num_constituents = num_constituents
        self.recovery_rate = recovery_rate

        if isinstance(day_count, str):
            day_count = _get_convention(day_count)
        self.day_count = day_count

        # Per-name notional
        self._per_name_notional = notional / num_constituents

        # Default count for index factor
        self._defaults = 0

    @property
    def index_factor(self) -> float:
        """Index factor = (N - defaults) / N."""
        return (self.num_constituents - self._defaults) / self.num_constituents

    def set_defaults(self, num_defaults: int):
        """Set number of defaulted names."""
        self._defaults = min(num_defaults, self.num_constituents)

    def present_value(
        self,
        valuation_date: date,
        discount_curve: Curve,
        survival_curve: SurvivalCurve,
    ) -> float:
        """
        Calculate index present value (simplified).

        Assumes homogeneous portfolio with same survival curve.
        """
        # Effective notional after defaults
        effective_notional = self.notional * self.index_factor

        # Create representative CDS
        cds = CreditDefaultSwap(
            notional=effective_notional,
            effective_date=self.effective_date,
            maturity_date=self.maturity_date,
            spread=self.spread,
            cds_type=CDSType.PROTECTION_BUYER,
            recovery_rate=self.recovery_rate,
            day_count=self.day_count,
        )

        return cds.present_value(valuation_date, discount_curve, survival_curve)

    def intrinsic_spread(
        self,
        valuation_date: date,
        discount_curve: Curve,
        constituent_spreads: List[float],
    ) -> float:
        """
        Calculate intrinsic (fair) index spread from constituents.

        Intrinsic = Σ wᵢ * Spreadᵢ

        where wᵢ is relative weight of constituent.
        """
        if len(constituent_spreads) != self.num_constituents:
            raise ValueError("Must provide spread for each constituent")

        # Simple average (assuming equal weights)
        return sum(constituent_spreads) / len(constituent_spreads)

    def __repr__(self) -> str:
        return (f"CDSIndex(notional={self.notional:,.0f}, "
                f"spread={self.spread * 10000:.0f}bp, "
                f"constituents={self.num_constituents}, "
                f"factor={self.index_factor:.2%})")
