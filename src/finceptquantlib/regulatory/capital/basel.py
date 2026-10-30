"""
Basel III/IV Capital Requirements

Implements Basel Committee capital adequacy framework:
- Credit Risk (Standardized and IRB)
- Market Risk
- Operational Risk
- Capital Ratios (CET1, Tier 1, Total Capital)
- Leverage Ratio
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import math


class AssetClass(Enum):
    """Basel asset classes for risk weighting."""
    SOVEREIGN = "sovereign"
    BANK = "bank"
    CORPORATE = "corporate"
    RETAIL = "retail"
    RESIDENTIAL_MORTGAGE = "residential_mortgage"
    COMMERCIAL_REAL_ESTATE = "commercial_real_estate"
    EQUITY = "equity"
    OTHER = "other"


@dataclass
class CreditExposure:
    """
    Credit exposure for RWA calculation.

    Attributes:
        exposure_amount: Exposure at default (EAD)
        asset_class: Type of exposure
        risk_weight: Supervisory risk weight (%)
        rating: Credit rating (optional)
        pd: Probability of default (for IRB)
        lgd: Loss given default (for IRB)
        maturity: Effective maturity in years
    """
    exposure_amount: float
    asset_class: AssetClass
    risk_weight: float = 100.0
    rating: Optional[str] = None
    pd: Optional[float] = None
    lgd: Optional[float] = None
    maturity: float = 2.5

    def rwa_standardized(self) -> float:
        """Calculate RWA using standardized approach."""
        return self.exposure_amount * (self.risk_weight / 100.0)

    def rwa_irb(self) -> float:
        """Calculate RWA using Internal Ratings-Based approach."""
        if self.pd is None or self.lgd is None:
            raise ValueError("PD and LGD required for IRB approach")

        if self.pd <= 0:
            return 0.0  # Zero default probability means zero risk weight

        # Basel II/III IRB correlation formula:
        # R = 0.12 * (1 - exp(-50*PD))/(1 - exp(-50)) + 0.24 * (1 - (1 - exp(-50*PD))/(1 - exp(-50)))
        exp_factor = (1 - math.exp(-50 * self.pd)) / (1 - math.exp(-50))
        correlation = 0.12 * exp_factor + 0.24 * (1 - exp_factor)

        # Normal distribution approximation
        from scipy.stats import norm
        k = self.lgd * norm.cdf(
            (norm.ppf(self.pd) + math.sqrt(correlation) * norm.ppf(0.999))
            / math.sqrt(1 - correlation)
        )

        # Maturity adjustment
        b = (0.11852 - 0.05478 * math.log(self.pd)) ** 2
        ma = (1 + (self.maturity - 2.5) * b) / (1 - 1.5 * b)

        rwa = self.exposure_amount * k * ma * 12.5  # 12.5 = 1/0.08
        return rwa


@dataclass
class RiskWeightedAssets:
    """Total Risk-Weighted Assets."""
    credit_risk_rwa: float = 0.0
    market_risk_rwa: float = 0.0
    operational_risk_rwa: float = 0.0

    @property
    def total_rwa(self) -> float:
        """Total RWA across all risk types."""
        return self.credit_risk_rwa + self.market_risk_rwa + self.operational_risk_rwa


class CreditRiskStandardized:
    """Credit risk calculation using Standardized Approach."""

    # Standard risk weights by rating (%)
    SOVEREIGN_RISK_WEIGHTS = {
        "AAA": 0, "AA+": 0, "AA": 0, "AA-": 0,
        "A+": 20, "A": 20, "A-": 20,
        "BBB+": 50, "BBB": 50, "BBB-": 50,
        "BB+": 100, "BB": 100, "BB-": 100,
        "B+": 100, "B": 100, "B-": 100,
        "CCC": 150, "Below CCC": 150,
        "Unrated": 100
    }

    CORPORATE_RISK_WEIGHTS = {
        "AAA": 20, "AA+": 20, "AA": 20, "AA-": 20,
        "A+": 50, "A": 50, "A-": 50,
        "BBB+": 100, "BBB": 100, "BBB-": 100,
        "BB+": 100, "BB": 100, "BB-": 100,
        "Below BB-": 150,
        "Unrated": 100
    }

    def __init__(self) -> None:
        """Initialize credit risk calculator."""
        self.exposures: List[CreditExposure] = []

    def add_exposure(self, exposure: CreditExposure) -> None:
        """Add credit exposure."""
        self.exposures.append(exposure)

    def calculate_rwa(self) -> float:
        """Calculate total credit RWA."""
        return sum(exp.rwa_standardized() for exp in self.exposures)


class CreditRiskIRB:
    """Credit risk using Internal Ratings-Based approach."""

    def __init__(self) -> None:
        """Initialize IRB calculator."""
        self.exposures: List[CreditExposure] = []

    def add_exposure(self, exposure: CreditExposure) -> None:
        """Add exposure with PD/LGD parameters."""
        if exposure.pd is None or exposure.lgd is None:
            raise ValueError("IRB requires PD and LGD")
        self.exposures.append(exposure)

    def calculate_rwa(self) -> float:
        """Calculate IRB RWA."""
        return sum(exp.rwa_irb() for exp in self.exposures)


class MarketRiskStandardized:
    """Market risk using standardized approach (pre-FRTB)."""

    def __init__(self) -> None:
        """Initialize market risk calculator."""
        self.interest_rate_risk = 0.0
        self.equity_risk = 0.0
        self.fx_risk = 0.0
        self.commodity_risk = 0.0

    def calculate_rwa(self) -> float:
        """Calculate market risk RWA."""
        # Market risk capital * 12.5
        total_capital = (self.interest_rate_risk + self.equity_risk +
                        self.fx_risk + self.commodity_risk)
        return total_capital * 12.5


class OperationalRisk:
    """Operational risk capital using Standardized Approach."""

    def __init__(self, business_indicator: float) -> None:
        """
        Initialize operational risk.

        Args:
            business_indicator: Business Indicator Component (BIC)
                               = Interest + Services + Financial income
        """
        self.business_indicator = business_indicator

    def calculate_rwa(self) -> float:
        """
        Calculate operational risk RWA.

        Uses tiered approach based on BIC size.
        """
        bi = self.business_indicator

        # Tiered marginal coefficients
        if bi <= 1_000_000_000:  # €1bn
            capital = bi * 0.12
        elif bi <= 30_000_000_000:  # €30bn
            capital = 1_000_000_000 * 0.12 + (bi - 1_000_000_000) * 0.15
        else:
            capital = 1_000_000_000 * 0.12 + 29_000_000_000 * 0.15 + (bi - 30_000_000_000) * 0.18

        return capital * 12.5


@dataclass
class CommonEquityTier1:
    """CET1 capital components."""
    common_shares: float = 0.0
    retained_earnings: float = 0.0
    accumulated_other_comprehensive_income: float = 0.0
    regulatory_adjustments: float = 0.0  # Deductions

    @property
    def total(self) -> float:
        """Total CET1 capital."""
        return (self.common_shares + self.retained_earnings +
                self.accumulated_other_comprehensive_income - self.regulatory_adjustments)


@dataclass
class Tier1Capital:
    """Total Tier 1 capital (CET1 + AT1)."""
    cet1: CommonEquityTier1
    additional_tier1: float = 0.0  # AT1 instruments

    @property
    def total(self) -> float:
        """Total Tier 1 capital."""
        return self.cet1.total + self.additional_tier1


@dataclass
class Tier2Capital:
    """Tier 2 capital instruments."""
    subordinated_debt: float = 0.0
    general_provisions: float = 0.0
    revaluation_reserves: float = 0.0

    @property
    def total(self) -> float:
        """Total Tier 2 capital."""
        return self.subordinated_debt + self.general_provisions + self.revaluation_reserves


@dataclass
class CapitalRatios:
    """Basel capital adequacy ratios."""
    cet1_ratio: float
    tier1_ratio: float
    total_capital_ratio: float
    leverage_ratio: float

    def is_adequately_capitalized(self) -> bool:
        """Check if all ratios meet minimum requirements."""
        return (self.cet1_ratio >= 4.5 and
                self.tier1_ratio >= 6.0 and
                self.total_capital_ratio >= 8.0 and
                self.leverage_ratio >= 3.0)

    def is_well_capitalized(self) -> bool:
        """Check if ratios exceed well-capitalized thresholds."""
        return (self.cet1_ratio >= 7.0 and
                self.tier1_ratio >= 8.5 and
                self.total_capital_ratio >= 10.5 and
                self.leverage_ratio >= 5.0)


@dataclass
class LeverageRatio:
    """Basel III Leverage Ratio."""
    tier1_capital: float
    total_exposure_measure: float

    @property
    def ratio(self) -> float:
        """Leverage ratio percentage."""
        if self.total_exposure_measure == 0:
            return 0.0
        return (self.tier1_capital / self.total_exposure_measure) * 100


class BaselCalculator:
    """
    Comprehensive Basel III/IV calculator.

    Calculates all required capital ratios and checks regulatory compliance.

    Examples:
        >>> basel = BaselCalculator()
        >>> basel.set_capital(cet1=500_000_000, tier1_additional=100_000_000, tier2=100_000_000)
        >>> basel.add_credit_exposure(CreditExposure(1_000_000_000, AssetClass.CORPORATE, 100))
        >>> ratios = basel.calculate_ratios()
        >>> print(f"CET1 Ratio: {ratios.cet1_ratio:.2f}%")
        CET1 Ratio: 50.00%
    """

    def __init__(self) -> None:
        """Initialize Basel calculator."""
        self.cet1 = CommonEquityTier1()
        self.tier1 = Tier1Capital(cet1=self.cet1)
        self.tier2 = Tier2Capital()
        self.credit_risk = CreditRiskStandardized()
        self.market_risk = MarketRiskStandardized()
        self.op_risk: Optional[OperationalRisk] = None
        self.total_assets = 0.0

    def set_capital(
        self,
        cet1: float = 0.0,
        tier1_additional: float = 0.0,
        tier2: float = 0.0,
        **cet1_components
    ) -> None:
        """Set capital levels."""
        if cet1_components:
            self.cet1 = CommonEquityTier1(**cet1_components)
        else:
            self.cet1 = CommonEquityTier1(common_shares=cet1)

        self.tier1 = Tier1Capital(cet1=self.cet1, additional_tier1=tier1_additional)
        self.tier2 = Tier2Capital(subordinated_debt=tier2)

    def add_credit_exposure(self, exposure: CreditExposure) -> None:
        """Add credit exposure for RWA calculation."""
        self.credit_risk.add_exposure(exposure)

    def set_operational_risk(self, business_indicator: float) -> None:
        """Set operational risk parameters."""
        self.op_risk = OperationalRisk(business_indicator)

    def set_total_assets(self, assets: float) -> None:
        """Set total assets for leverage ratio."""
        self.total_assets = assets

    def calculate_rwa(self) -> RiskWeightedAssets:
        """Calculate total RWA."""
        credit_rwa = self.credit_risk.calculate_rwa()
        market_rwa = self.market_risk.calculate_rwa()
        op_rwa = self.op_risk.calculate_rwa() if self.op_risk else 0.0

        return RiskWeightedAssets(
            credit_risk_rwa=credit_rwa,
            market_risk_rwa=market_rwa,
            operational_risk_rwa=op_rwa
        )

    def calculate_ratios(self) -> CapitalRatios:
        """Calculate all Basel capital ratios."""
        rwa = self.calculate_rwa()
        total_rwa = rwa.total_rwa

        if total_rwa == 0:
            return CapitalRatios(0, 0, 0, 0)

        cet1_ratio = (self.cet1.total / total_rwa) * 100
        tier1_ratio = (self.tier1.total / total_rwa) * 100
        total_capital_ratio = ((self.tier1.total + self.tier2.total) / total_rwa) * 100

        leverage_ratio = 0.0
        if self.total_assets > 0:
            leverage_ratio = (self.tier1.total / self.total_assets) * 100

        return CapitalRatios(
            cet1_ratio=cet1_ratio,
            tier1_ratio=tier1_ratio,
            total_capital_ratio=total_capital_ratio,
            leverage_ratio=leverage_ratio
        )


def calculate_cet1_ratio(cet1_capital: float, rwa: float) -> float:
    """Calculate CET1 ratio."""
    return (cet1_capital / rwa) * 100 if rwa > 0 else 0.0


def calculate_tier1_ratio(tier1_capital: float, rwa: float) -> float:
    """Calculate Tier 1 ratio."""
    return (tier1_capital / rwa) * 100 if rwa > 0 else 0.0


def calculate_total_capital_ratio(total_capital: float, rwa: float) -> float:
    """Calculate total capital ratio."""
    return (total_capital / rwa) * 100 if rwa > 0 else 0.0


def calculate_leverage_ratio(tier1_capital: float, total_exposure: float) -> float:
    """Calculate leverage ratio."""
    return (tier1_capital / total_exposure) * 100 if total_exposure > 0 else 0.0
