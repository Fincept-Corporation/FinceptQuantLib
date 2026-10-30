"""
Basel III Liquidity Ratios

Implements liquidity coverage requirements:
- LCR (Liquidity Coverage Ratio)
- NSFR (Net Stable Funding Ratio)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class HQLALevel(Enum):
    """High-Quality Liquid Assets levels."""
    LEVEL_1 = 1  # 100% weighting (cash, central bank reserves)
    LEVEL_2A = 2  # 85% weighting (high-quality sovereigns, covered bonds)
    LEVEL_2B = 3  # 50% weighting (lower-rated sovereigns, corporate bonds)


@dataclass
class HighQualityLiquidAssets:
    """
    HQLA components for LCR.

    Level 1: Cash, central bank reserves, sovereign debt
    Level 2A: High-quality covered bonds, corporate bonds
    Level 2B: Lower-rated securities (subject to caps)
    """
    level_1: float = 0.0
    level_2a: float = 0.0
    level_2b: float = 0.0

    @property
    def total_weighted(self) -> float:
        """Total HQLA after haircuts."""
        # Level 2A: 85% weighting
        # Level 2B: 50% weighting, max 15% of total HQLA
        level_2a_adj = self.level_2a * 0.85
        level_2b_adj = min(self.level_2b * 0.50, (self.level_1 + level_2a_adj) * (15 / 85))  # 15/(1-0.15)

        return self.level_1 + level_2a_adj + level_2b_adj


@dataclass
class CashOutflows:
    """
    30-day net cash outflows for LCR.

    Attributes:
        retail_deposits_stable: Stable retail deposits (5% outflow)
        retail_deposits_less_stable: Less stable retail (10% outflow)
        unsecured_wholesale: Wholesale funding (various rates)
        secured_funding: Secured funding outflows
        derivatives_outflows: Net derivative cash flows
        credit_facilities: Committed credit/liquidity facilities
        other_contractual: Other contractual obligations
    """
    retail_deposits_stable: float = 0.0
    retail_deposits_less_stable: float = 0.0
    unsecured_wholesale: float = 0.0
    secured_funding: float = 0.0
    derivatives_outflows: float = 0.0
    credit_facilities: float = 0.0
    other_contractual: float = 0.0

    def total_outflows(self) -> float:
        """Calculate total expected cash outflows."""
        outflows = (
            self.retail_deposits_stable * 0.05 +
            self.retail_deposits_less_stable * 0.10 +
            self.unsecured_wholesale * 1.00 +  # 100% for most wholesale
            self.secured_funding * 0.25 +
            self.derivatives_outflows +
            self.credit_facilities * 0.05 +  # 5% for committed facilities
            self.other_contractual
        )
        return outflows


@dataclass
class CashInflows:
    """
    30-day expected cash inflows for LCR.

    Capped at 75% of outflows per Basel III.
    """
    secured_lending: float = 0.0
    committed_facilities: float = 0.0
    other_inflows: float = 0.0

    def total_inflows(self, max_cap_pct: float = 0.75) -> float:
        """Calculate total inflows (subject to cap)."""
        raw_inflows = (
            self.secured_lending * 1.00 +
            self.committed_facilities * 0.00 +  # Not counted
            self.other_inflows
        )
        return raw_inflows  # Cap applied in LCR calculation


@dataclass
class LiquidityRatios:
    """Liquidity ratio results."""
    lcr: float
    nsfr: float
    hqla: float
    net_outflows: float
    asf: float
    rsf: float

    def is_compliant(self) -> bool:
        """Check if ratios meet minimum requirements."""
        return self.lcr >= 100.0 and self.nsfr >= 100.0


class LCRCalculator:
    """
    Liquidity Coverage Ratio calculator.

    LCR = HQLA / Net Cash Outflows (30 days)
    Minimum requirement: 100%

    Examples:
        >>> hqla = HighQualityLiquidAssets(level_1=500_000_000, level_2a=200_000_000)
        >>> outflows = CashOutflows(retail_deposits_stable=2_000_000_000,
        ...                         unsecured_wholesale=1_000_000_000)
        >>> inflows = CashInflows(other_inflows=100_000_000)
        >>> lcr_calc = LCRCalculator(hqla, outflows, inflows)
        >>> ratio = lcr_calc.calculate()
        >>> print(f"LCR: {ratio:.2f}%")
        LCR: 67.00%
    """

    def __init__(
        self,
        hqla: HighQualityLiquidAssets,
        outflows: CashOutflows,
        inflows: CashInflows
    ) -> None:
        """Initialize LCR calculator."""
        self.hqla = hqla
        self.outflows = outflows
        self.inflows = inflows

    def calculate(self) -> float:
        """
        Calculate LCR percentage.

        Returns:
            LCR ratio (e.g., 120.5 for 120.5%)
        """
        total_hqla = self.hqla.total_weighted
        total_outflows = self.outflows.total_outflows()

        # Inflows capped at 75% of outflows
        total_inflows = self.inflows.total_inflows()
        capped_inflows = min(total_inflows, 0.75 * total_outflows)

        net_outflows = max(total_outflows - capped_inflows, 0)

        if net_outflows == 0:
            return float('inf')

        return (total_hqla / net_outflows) * 100


@dataclass
class AvailableStableFunding:
    """
    Available Stable Funding for NSFR.

    Weighted by stability of funding source.
    """
    capital: float = 0.0  # 100% ASF factor
    stable_deposits_retail: float = 0.0  # 95%
    less_stable_deposits_retail: float = 0.0  # 90%
    stable_deposits_wholesale: float = 0.0  # 50%
    less_stable_deposits_wholesale: float = 0.0  # 0%
    other_liabilities: float = 0.0  # 0%

    def total_asf(self) -> float:
        """Calculate total available stable funding."""
        return (
            self.capital * 1.00 +
            self.stable_deposits_retail * 0.95 +
            self.less_stable_deposits_retail * 0.90 +
            self.stable_deposits_wholesale * 0.50 +
            self.less_stable_deposits_wholesale * 0.00 +
            self.other_liabilities * 0.00
        )


@dataclass
class RequiredStableFunding:
    """
    Required Stable Funding for NSFR.

    Weighted by liquidity of assets.
    """
    cash_and_reserves: float = 0.0  # 0% RSF
    securities_level1: float = 0.0  # 5%
    securities_level2a: float = 0.0  # 15%
    securities_level2b: float = 0.0  # 50%
    loans_retail: float = 0.0  # 65-85%
    loans_corporate: float = 0.0  # 85%
    other_assets: float = 0.0  # 85%
    unencumbered_performing_loans: float = 0.0  # 65%

    def total_rsf(self) -> float:
        """Calculate total required stable funding."""
        return (
            self.cash_and_reserves * 0.00 +
            self.securities_level1 * 0.05 +
            self.securities_level2a * 0.15 +
            self.securities_level2b * 0.50 +
            self.loans_retail * 0.85 +
            self.loans_corporate * 0.85 +
            self.unencumbered_performing_loans * 0.65 +
            self.other_assets * 0.85
        )


class NSFRCalculator:
    """
    Net Stable Funding Ratio calculator.

    NSFR = Available Stable Funding / Required Stable Funding
    Minimum requirement: 100%

    Examples:
        >>> asf = AvailableStableFunding(capital=500_000_000,
        ...                              stable_deposits_retail=3_000_000_000)
        >>> rsf = RequiredStableFunding(loans_retail=2_000_000_000,
        ...                             loans_corporate=1_500_000_000)
        >>> nsfr_calc = NSFRCalculator(asf, rsf)
        >>> ratio = nsfr_calc.calculate()
        >>> print(f"NSFR: {ratio:.2f}%")
        NSFR: 112.61%
    """

    def __init__(
        self,
        asf: AvailableStableFunding,
        rsf: RequiredStableFunding
    ) -> None:
        """Initialize NSFR calculator."""
        self.asf = asf
        self.rsf = rsf

    def calculate(self) -> float:
        """
        Calculate NSFR percentage.

        Returns:
            NSFR ratio (e.g., 110.5 for 110.5%)
        """
        total_asf = self.asf.total_asf()
        total_rsf = self.rsf.total_rsf()

        if total_rsf == 0:
            return float('inf')

        return (total_asf / total_rsf) * 100


def calculate_lcr(hqla: float, net_outflows: float) -> float:
    """Simple LCR calculation."""
    return (hqla / net_outflows) * 100 if net_outflows > 0 else float('inf')


def calculate_nsfr(asf: float, rsf: float) -> float:
    """Simple NSFR calculation."""
    return (asf / rsf) * 100 if rsf > 0 else float('inf')
