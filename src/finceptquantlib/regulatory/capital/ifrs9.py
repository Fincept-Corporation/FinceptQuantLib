"""
IFRS 9 / CECL Expected Credit Loss Modeling

Implements expected credit loss calculation under:
- IFRS 9 (International Financial Reporting Standards)
- CECL (Current Expected Credit Loss - US GAAP)

Key concepts:
- Stage 1: 12-month ECL for performing loans
- Stage 2: Lifetime ECL for significant increase in credit risk
- Stage 3: Lifetime ECL for credit-impaired assets
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple


class Stage(Enum):
    """IFRS 9 staging classification."""
    STAGE_1 = 1  # 12-month ECL
    STAGE_2 = 2  # Lifetime ECL (not credit-impaired)
    STAGE_3 = 3  # Lifetime ECL (credit-impaired)


@dataclass
class ProbabilityOfDefault:
    """
    PD term structure for ECL calculation.

    Attributes:
        periods: Time periods (in years)
        pds: Point-in-time PD for each period
        cumulative: Cumulative default probabilities
    """
    periods: List[float]
    pds: List[float]

    @property
    def cumulative(self) -> List[float]:
        """Calculate cumulative default probabilities."""
        cum_pds = []
        survival = 1.0
        for pd in self.pds:
            survival *= (1 - pd)
            cum_pds.append(1 - survival)
        return cum_pds

    def marginal_pd(self, period_index: int) -> float:
        """Marginal PD for a specific period."""
        if period_index >= len(self.pds):
            return 0.0
        return self.pds[period_index]


@dataclass
class LossGivenDefault:
    """
    Loss given default parameters.

    Attributes:
        lgd_rate: LGD percentage (0-1)
        collateral_value: Collateral recovery value
        exposure_at_default: Exposure amount at default
    """
    lgd_rate: float
    collateral_value: float = 0.0
    exposure_at_default: float = 0.0

    def effective_lgd(self) -> float:
        """Calculate effective LGD after collateral."""
        if self.exposure_at_default == 0:
            return self.lgd_rate

        recovery_rate = min(self.collateral_value / self.exposure_at_default, 1.0)
        return max(self.lgd_rate - recovery_rate, 0.0)


@dataclass
class ExposureAtDefault:
    """
    Exposure at default projection.

    Attributes:
        initial_exposure: Current exposure
        periods: Future periods
        projected_exposures: Projected EAD over time (considers amortization, prepayment)
        discount_rates: Discount rates for present value
    """
    initial_exposure: float
    periods: List[float]
    projected_exposures: List[float]
    discount_rates: List[float]

    def discounted_exposure(self, period_index: int) -> float:
        """Get discounted exposure for a period."""
        if period_index >= len(self.projected_exposures):
            return 0.0

        exposure = self.projected_exposures[period_index]
        discount_rate = self.discount_rates[period_index] if period_index < len(self.discount_rates) else 0.0
        period = self.periods[period_index]

        return exposure / ((1 + discount_rate) ** period)


@dataclass
class ExpectedCreditLoss:
    """
    Complete ECL calculation result.

    ECL = Sum over periods: PD(t) * LGD(t) * EAD(t) * DF(t)
    """
    stage: Stage
    twelve_month_ecl: float
    lifetime_ecl: float
    applicable_ecl: float  # 12M for Stage 1, Lifetime for Stage 2/3

    @property
    def provision_amount(self) -> float:
        """ECL provision to book."""
        return self.applicable_ecl


@dataclass
class StageAssessment:
    """Assessment of staging classification."""
    current_stage: Stage
    previous_stage: Stage
    significant_increase: bool
    credit_impaired: bool
    days_past_due: int
    rating_downgrade_notches: int


class ECLCalculator:
    """
    Expected Credit Loss calculator for IFRS 9 / CECL.

    Calculates ECL using PD, LGD, and EAD term structures.

    Examples:
        >>> pd = ProbabilityOfDefault([1, 2, 3], [0.01, 0.015, 0.02])
        >>> lgd = LossGivenDefault(lgd_rate=0.45)
        >>> ead = ExposureAtDefault(1_000_000, [1, 2, 3], [950_000, 900_000, 850_000], [0.05]*3)
        >>> ecl_calc = ECLCalculator(pd, lgd, ead)
        >>> result = ecl_calc.calculate(Stage.STAGE_1)
        >>> print(f"ECL: ${result.provision_amount:,.2f}")
        ECL: $4,071.43
    """

    def __init__(
        self,
        pd: ProbabilityOfDefault,
        lgd: LossGivenDefault,
        ead: ExposureAtDefault,
        discount_rate: float = 0.05
    ) -> None:
        """
        Initialize ECL calculator.

        Args:
            pd: Probability of default term structure
            lgd: Loss given default parameters
            ead: Exposure at default projection
            discount_rate: Effective interest rate for discounting
        """
        self.pd = pd
        self.lgd = lgd
        self.ead = ead
        self.discount_rate = discount_rate

    def calculate(self, stage: Stage) -> ExpectedCreditLoss:
        """
        Calculate ECL for given stage.

        Args:
            stage: IFRS 9 stage classification

        Returns:
            ExpectedCreditLoss with 12M and lifetime calculations
        """
        # 12-month ECL (Stage 1)
        twelve_month = self._calculate_period_ecl(periods=1)

        # Lifetime ECL (Stages 2 & 3)
        lifetime = self._calculate_period_ecl(periods=len(self.pd.periods))

        # Applicable ECL based on stage
        applicable = twelve_month if stage == Stage.STAGE_1 else lifetime

        return ExpectedCreditLoss(
            stage=stage,
            twelve_month_ecl=twelve_month,
            lifetime_ecl=lifetime,
            applicable_ecl=applicable
        )

    def _calculate_period_ecl(self, periods: int) -> float:
        """Calculate ECL for specified number of periods."""
        total_ecl = 0.0
        effective_lgd = self.lgd.effective_lgd()

        for i in range(min(periods, len(self.pd.periods))):
            pd_marginal = self.pd.marginal_pd(i)
            exposure = self.ead.discounted_exposure(i)

            # ECL = PD * LGD * EAD
            period_ecl = pd_marginal * effective_lgd * exposure
            total_ecl += period_ecl

        return total_ecl


def twelve_month_ecl(
    pd_12m: float,
    lgd: float,
    ead: float,
    discount_rate: float = 0.0
) -> float:
    """
    Simple 12-month ECL calculation.

    Args:
        pd_12m: 12-month probability of default
        lgd: Loss given default rate
        ead: Exposure at default
        discount_rate: Discount rate for PV

    Returns:
        12-month expected credit loss
    """
    ecl = pd_12m * lgd * ead
    if discount_rate > 0:
        ecl = ecl / (1 + discount_rate)
    return ecl


def lifetime_ecl(
    pd_curve: List[float],
    lgd: float,
    ead_curve: List[float],
    discount_rates: List[float]
) -> float:
    """
    Lifetime ECL calculation.

    Args:
        pd_curve: PD for each period
        lgd: Loss given default
        ead_curve: EAD for each period
        discount_rates: Discount rates for each period

    Returns:
        Lifetime expected credit loss (present value)
    """
    total_ecl = 0.0
    survival = 1.0

    for i, pd in enumerate(pd_curve):
        # Marginal PD for period
        marginal_pd = survival * pd
        survival *= (1 - pd)

        # Period ECL
        ead = ead_curve[i] if i < len(ead_curve) else ead_curve[-1]
        dr = discount_rates[i] if i < len(discount_rates) else discount_rates[-1]

        period_ecl = marginal_pd * lgd * ead / ((1 + dr) ** (i + 1))
        total_ecl += period_ecl

    return total_ecl


def assess_stage(
    days_past_due: int,
    rating_downgrade: int,
    pd_increase_ratio: float,
    is_credit_impaired: bool = False
) -> Stage:
    """
    Assess IFRS 9 stage classification.

    Args:
        days_past_due: Days payment is overdue
        rating_downgrade: Number of notches rating has decreased
        pd_increase_ratio: Current PD / Origination PD
        is_credit_impaired: Whether asset is credit-impaired

    Returns:
        Stage classification
    """
    # Stage 3: Credit-impaired
    if is_credit_impaired or days_past_due >= 90:
        return Stage.STAGE_3

    # Stage 2: Significant increase in credit risk
    if (days_past_due >= 30 or
        rating_downgrade >= 3 or
        pd_increase_ratio >= 2.0):
        return Stage.STAGE_2

    # Stage 1: Performing
    return Stage.STAGE_1


def significant_increase_in_credit_risk(
    origination_pd: float,
    current_pd: float,
    threshold_absolute: float = 0.005,
    threshold_relative: float = 2.0
) -> bool:
    """
    Determine if there's been a significant increase in credit risk.

    Args:
        origination_pd: PD at loan origination
        current_pd: Current PD
        threshold_absolute: Absolute PD increase threshold
        threshold_relative: Relative PD increase multiple

    Returns:
        True if significant increase detected
    """
    absolute_increase = current_pd - origination_pd
    relative_increase = current_pd / origination_pd if origination_pd > 0 else float('inf')

    return (absolute_increase >= threshold_absolute or
            relative_increase >= threshold_relative)


class IFRS9Calculator:
    """
    Complete IFRS 9 impairment calculator.

    Manages staging, ECL calculation, and provision updates.
    """

    def __init__(self) -> None:
        """Initialize IFRS 9 calculator."""
        self.exposures: List[Tuple[str, ECLCalculator, Stage]] = []

    def add_exposure(
        self,
        exposure_id: str,
        pd: ProbabilityOfDefault,
        lgd: LossGivenDefault,
        ead: ExposureAtDefault,
        stage: Stage
    ) -> None:
        """Add exposure for ECL calculation."""
        ecl_calc = ECLCalculator(pd, lgd, ead)
        self.exposures.append((exposure_id, ecl_calc, stage))

    def calculate_total_provision(self) -> float:
        """Calculate total ECL provision across all exposures."""
        total = 0.0
        for _, ecl_calc, stage in self.exposures:
            result = ecl_calc.calculate(stage)
            total += result.provision_amount
        return total

    def get_provision_by_stage(self) -> Dict[Stage, float]:
        """Get ECL provisions grouped by stage."""
        provisions = {Stage.STAGE_1: 0.0, Stage.STAGE_2: 0.0, Stage.STAGE_3: 0.0}

        for _, ecl_calc, stage in self.exposures:
            result = ecl_calc.calculate(stage)
            provisions[stage] += result.provision_amount

        return provisions
