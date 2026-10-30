"""
Real Estate Investment Trust (REIT) Analysis

Specialized financial metrics for REITs:
- Funds From Operations (FFO) and Adjusted FFO
- Net Asset Value (NAV) Analysis
- Valuation Metrics (P/FFO, P/NAV, Cap Rates)
- Dividend Analysis
- Property Operating Metrics
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


@dataclass
class REITFinancials:
    """Financial data structure for Real Estate Investment Trusts."""

    # Income Statement
    total_revenues: float = 0.0
    rental_revenue: float = 0.0
    other_revenue: float = 0.0
    property_operating_expenses: float = 0.0
    general_admin_expenses: float = 0.0
    interest_expense: float = 0.0
    depreciation_amortization: float = 0.0
    net_income: float = 0.0

    # FFO Adjustments
    gains_on_property_sales: float = 0.0
    impairments: float = 0.0
    depreciation_real_estate: float = 0.0

    # AFFO Adjustments
    straight_line_rent_adjustment: float = 0.0
    recurring_capex: float = 0.0
    tenant_improvements: float = 0.0
    leasing_commissions: float = 0.0
    amortization_above_below_market_leases: float = 0.0

    # Balance Sheet
    total_assets: float = 0.0
    real_estate_assets_gross: float = 0.0
    accumulated_depreciation: float = 0.0
    real_estate_assets_net: float = 0.0
    total_debt: float = 0.0
    total_liabilities: float = 0.0
    total_equity: float = 0.0

    # Property Data
    total_square_feet: float = 0.0
    occupied_square_feet: float = 0.0
    number_of_properties: int = 0

    # Market Data
    stock_price: float = 0.0
    shares_outstanding: float = 0.0
    dividends_per_share: float = 0.0

    # Same-Store Data (for growth analysis)
    same_store_noi_current: float = 0.0
    same_store_noi_prior: float = 0.0

    # Lease Data
    weighted_average_lease_expiry: float = 0.0  # In years
    average_rent_per_sqft: float = 0.0
    market_rent_per_sqft: float = 0.0

    @property
    def net_operating_income(self) -> float:
        """Net Operating Income (NOI)."""
        return self.rental_revenue + self.other_revenue - self.property_operating_expenses

    @property
    def market_cap(self) -> float:
        """Market capitalization."""
        return self.stock_price * self.shares_outstanding

    @property
    def enterprise_value(self) -> float:
        """Enterprise value."""
        return self.market_cap + self.total_debt - self.total_assets * 0.02  # Approx cash

    @property
    def occupancy_rate(self) -> Optional[float]:
        """Occupancy rate."""
        if self.total_square_feet <= 0:
            return None
        return self.occupied_square_feet / self.total_square_feet


# =============================================================================
# FFO and AFFO Calculations
# =============================================================================

def funds_from_operations(
    net_income: float,
    depreciation_real_estate: float,
    gains_on_sales: float = 0.0,
    impairments: float = 0.0
) -> float:
    """
    Funds From Operations (FFO) - NAREIT Definition.

    Primary earnings measure for REITs.

    FFO = Net Income + Real Estate Depreciation - Gains on Sales + Impairments

    Args:
        net_income: GAAP net income
        depreciation_real_estate: Depreciation on real estate assets
        gains_on_sales: Gains/losses on property dispositions
        impairments: Real estate impairment charges

    Returns:
        FFO amount
    """
    return net_income + depreciation_real_estate - gains_on_sales + impairments


def adjusted_ffo(
    ffo: float,
    straight_line_rent: float = 0.0,
    recurring_capex: float = 0.0,
    tenant_improvements: float = 0.0,
    leasing_commissions: float = 0.0,
    above_below_market_lease_amort: float = 0.0
) -> float:
    """
    Adjusted Funds From Operations (AFFO).

    Also called Cash Available for Distribution (CAD) or Funds Available
    for Distribution (FAD).

    AFFO = FFO - Recurring CapEx - TIs - LCs - Straight-Line Rent Adj
           - Above/Below Market Lease Amortization

    Args:
        ffo: Funds from operations
        straight_line_rent: Non-cash straight-line rent adjustment
        recurring_capex: Normalized maintenance capital expenditures
        tenant_improvements: Tenant improvement costs
        leasing_commissions: Leasing commission expenses
        above_below_market_lease_amort: Amortization of lease intangibles

    Returns:
        AFFO amount
    """
    return (ffo - straight_line_rent - recurring_capex -
            tenant_improvements - leasing_commissions -
            above_below_market_lease_amort)


def cash_available_for_distribution(
    affo: float,
    principal_repayments: float = 0.0,
    other_adjustments: float = 0.0
) -> float:
    """
    Cash Available for Distribution (CAD).

    More conservative than AFFO, includes debt repayment.

    Args:
        affo: Adjusted FFO
        principal_repayments: Scheduled debt principal payments
        other_adjustments: Other cash flow adjustments

    Returns:
        CAD amount
    """
    return affo - principal_repayments + other_adjustments


def ffo_per_share(ffo: float, shares_outstanding: float) -> Optional[float]:
    """FFO per share."""
    if shares_outstanding <= 0:
        return None
    return ffo / shares_outstanding


def affo_per_share(affo: float, shares_outstanding: float) -> Optional[float]:
    """AFFO per share."""
    if shares_outstanding <= 0:
        return None
    return affo / shares_outstanding


# =============================================================================
# NAV Calculations
# =============================================================================

def net_asset_value(
    real_estate_value: float,
    other_assets: float,
    total_liabilities: float
) -> float:
    """
    Net Asset Value (NAV).

    Estimated fair value of REIT equity.

    NAV = Fair Value of Real Estate + Other Assets - Total Liabilities

    Args:
        real_estate_value: Estimated fair value of properties
        other_assets: Other assets (cash, receivables, etc.)
        total_liabilities: Total liabilities

    Returns:
        NAV amount
    """
    return real_estate_value + other_assets - total_liabilities


def nav_from_cap_rate(
    noi: float,
    cap_rate: float,
    other_assets: float = 0.0,
    total_liabilities: float = 0.0
) -> Optional[float]:
    """
    Calculate NAV by capitalizing NOI.

    NAV = (NOI / Cap Rate) + Other Assets - Liabilities

    Args:
        noi: Net operating income
        cap_rate: Capitalization rate
        other_assets: Non-real estate assets
        total_liabilities: Total liabilities

    Returns:
        NAV amount
    """
    if cap_rate <= 0:
        return None
    property_value = noi / cap_rate
    return property_value + other_assets - total_liabilities


def nav_per_share(nav: float, shares_outstanding: float) -> Optional[float]:
    """NAV per share."""
    if shares_outstanding <= 0:
        return None
    return nav / shares_outstanding


# =============================================================================
# Valuation Metrics
# =============================================================================

def price_to_ffo(
    stock_price: float,
    ffo_per_share: float
) -> Optional[float]:
    """
    Price to FFO Ratio.

    Primary valuation metric for REITs (analogous to P/E).

    Typical ranges vary by sector:
    - Office: 10x - 15x
    - Retail: 10x - 14x
    - Industrial: 15x - 25x
    - Apartments: 15x - 22x

    Args:
        stock_price: Current stock price
        ffo_per_share: FFO per share

    Returns:
        P/FFO multiple
    """
    if ffo_per_share is None or ffo_per_share <= 0:
        return None
    return stock_price / ffo_per_share


def price_to_affo(
    stock_price: float,
    affo_per_share: float
) -> Optional[float]:
    """
    Price to AFFO Ratio.

    More conservative valuation metric.

    Args:
        stock_price: Current stock price
        affo_per_share: AFFO per share

    Returns:
        P/AFFO multiple
    """
    if affo_per_share is None or affo_per_share <= 0:
        return None
    return stock_price / affo_per_share


def price_to_nav(
    stock_price: float,
    nav_per_share: float
) -> Optional[float]:
    """
    Price to NAV Ratio.

    Measures premium/discount to estimated asset value.

    Interpretation:
    - > 1.0: Trading at premium to NAV
    - = 1.0: Trading at NAV
    - < 1.0: Trading at discount to NAV

    Args:
        stock_price: Current stock price
        nav_per_share: NAV per share

    Returns:
        P/NAV ratio
    """
    if nav_per_share is None or nav_per_share <= 0:
        return None
    return stock_price / nav_per_share


def implied_cap_rate(
    noi: float,
    enterprise_value: float
) -> Optional[float]:
    """
    Implied Capitalization Rate.

    Market-implied cap rate based on current valuation.

    Implied Cap = NOI / Enterprise Value

    Args:
        noi: Net operating income
        enterprise_value: Enterprise value (market cap + debt - cash)

    Returns:
        Implied cap rate as decimal
    """
    if enterprise_value <= 0:
        return None
    return noi / enterprise_value


def spread_to_treasuries(
    implied_cap_rate: float,
    treasury_yield: float
) -> float:
    """
    Cap Rate Spread to Treasuries.

    Risk premium over risk-free rate.

    Args:
        implied_cap_rate: Implied cap rate
        treasury_yield: 10-year treasury yield

    Returns:
        Spread in decimal
    """
    return implied_cap_rate - treasury_yield


# =============================================================================
# Dividend Metrics
# =============================================================================

def dividend_payout_ffo(
    dividends: float,
    ffo: float
) -> Optional[float]:
    """
    Dividend Payout Ratio (FFO basis).

    Dividend Payout = Dividends / FFO

    Typical ranges:
    - Conservative: < 70%
    - Moderate: 70% - 85%
    - Aggressive: > 85%

    REITs must pay 90%+ of taxable income, but FFO payout can be lower.

    Args:
        dividends: Total dividends paid
        ffo: Funds from operations

    Returns:
        Payout ratio as decimal
    """
    if ffo <= 0:
        return None
    return dividends / ffo


def dividend_payout_affo(
    dividends: float,
    affo: float
) -> Optional[float]:
    """
    Dividend Payout Ratio (AFFO basis).

    More conservative payout metric.

    Args:
        dividends: Total dividends paid
        affo: Adjusted funds from operations

    Returns:
        Payout ratio as decimal
    """
    if affo <= 0:
        return None
    return dividends / affo


def dividend_yield_reit(
    dividends_per_share: float,
    stock_price: float
) -> Optional[float]:
    """
    Dividend Yield.

    Args:
        dividends_per_share: Annual dividends per share
        stock_price: Current stock price

    Returns:
        Yield as decimal
    """
    if stock_price <= 0:
        return None
    return dividends_per_share / stock_price


# =============================================================================
# Leverage Metrics
# =============================================================================

def debt_to_ebitda_reit(
    total_debt: float,
    ebitda: float
) -> Optional[float]:
    """
    Debt to EBITDA Ratio.

    Key leverage metric for REITs.

    Typical ranges:
    - Conservative: < 5.0x
    - Moderate: 5.0x - 7.0x
    - Aggressive: > 7.0x

    Args:
        total_debt: Total debt outstanding
        ebitda: Earnings before interest, taxes, depreciation, amortization

    Returns:
        Ratio as multiple
    """
    if ebitda <= 0:
        return None
    return total_debt / ebitda


def debt_to_gross_assets(
    total_debt: float,
    gross_real_estate: float,
    other_assets: float = 0.0
) -> Optional[float]:
    """
    Debt to Gross Assets.

    Leverage measure using gross (undepreciated) assets.

    Typical covenant level: < 60%

    Args:
        total_debt: Total debt
        gross_real_estate: Gross real estate assets (before depreciation)
        other_assets: Other assets

    Returns:
        Ratio as decimal
    """
    total_gross = gross_real_estate + other_assets
    if total_gross <= 0:
        return None
    return total_debt / total_gross


def fixed_charge_coverage_reit(
    ebitda: float,
    interest_expense: float,
    preferred_dividends: float = 0.0,
    scheduled_principal: float = 0.0
) -> Optional[float]:
    """
    Fixed Charge Coverage Ratio.

    Ability to meet fixed obligations.

    FCCR = EBITDA / (Interest + Preferred Dividends + Principal)

    Typical covenant level: > 1.5x

    Args:
        ebitda: EBITDA
        interest_expense: Interest expense
        preferred_dividends: Preferred stock dividends
        scheduled_principal: Scheduled principal repayments

    Returns:
        Coverage ratio
    """
    fixed_charges = interest_expense + preferred_dividends + scheduled_principal
    if fixed_charges <= 0:
        return None
    return ebitda / fixed_charges


def interest_coverage_reit(
    ebitda: float,
    interest_expense: float
) -> Optional[float]:
    """
    Interest Coverage Ratio.

    EBITDA / Interest Expense

    Args:
        ebitda: EBITDA
        interest_expense: Interest expense

    Returns:
        Coverage ratio
    """
    if interest_expense <= 0:
        return None
    return ebitda / interest_expense


# =============================================================================
# Property Operating Metrics
# =============================================================================

def occupancy_rate_analysis(
    occupied_sqft: float,
    total_sqft: float
) -> Optional[float]:
    """
    Occupancy Rate.

    Occupancy = Occupied SF / Total SF

    Args:
        occupied_sqft: Occupied square feet
        total_sqft: Total leasable square feet

    Returns:
        Occupancy as decimal
    """
    if total_sqft <= 0:
        return None
    return occupied_sqft / total_sqft


def same_store_noi_growth(
    current_noi: float,
    prior_noi: float
) -> Optional[float]:
    """
    Same-Store NOI Growth.

    Key organic growth metric (excludes acquisitions/dispositions).

    Args:
        current_noi: Current period same-store NOI
        prior_noi: Prior period same-store NOI

    Returns:
        Growth rate as decimal
    """
    if prior_noi <= 0:
        return None
    return (current_noi - prior_noi) / prior_noi


def rent_spread(
    new_lease_rent: float,
    expiring_lease_rent: float
) -> Optional[float]:
    """
    Rent Spread on Renewals/New Leases.

    Measures pricing power.

    Spread = (New Rent - Expiring Rent) / Expiring Rent

    Args:
        new_lease_rent: Rent on new/renewed lease
        expiring_lease_rent: Rent on expiring lease

    Returns:
        Spread as decimal
    """
    if expiring_lease_rent <= 0:
        return None
    return (new_lease_rent - expiring_lease_rent) / expiring_lease_rent


def mark_to_market(
    in_place_rent: float,
    market_rent: float
) -> Optional[float]:
    """
    Mark-to-Market Rent Analysis.

    Embedded rent growth potential.

    MTM = (Market Rent - In-Place Rent) / In-Place Rent

    Positive = rents below market (upside)
    Negative = rents above market (risk)

    Args:
        in_place_rent: Current average in-place rent
        market_rent: Current market rent

    Returns:
        Mark-to-market as decimal
    """
    if in_place_rent <= 0:
        return None
    return (market_rent - in_place_rent) / in_place_rent


def weighted_average_lease_term(
    leases: List[Tuple[float, float]]
) -> Optional[float]:
    """
    Weighted Average Lease Term (WALT).

    Measure of lease duration and rollover risk.

    Args:
        leases: List of (annual_rent, remaining_term_years) tuples

    Returns:
        WALT in years
    """
    if not leases:
        return None

    total_rent = sum(rent for rent, _ in leases)
    if total_rent <= 0:
        return None

    weighted_term = sum(rent * term for rent, term in leases)
    return weighted_term / total_rent


def noi_margin(
    noi: float,
    total_revenues: float
) -> Optional[float]:
    """
    NOI Margin.

    Property-level operating efficiency.

    NOI Margin = NOI / Total Revenues

    Args:
        noi: Net operating income
        total_revenues: Total property revenues

    Returns:
        Margin as decimal
    """
    if total_revenues <= 0:
        return None
    return noi / total_revenues


# =============================================================================
# REIT Analysis Class
# =============================================================================

@dataclass
class FFOMetrics:
    """FFO analysis summary."""
    ffo: float
    affo: float
    ffo_per_share: Optional[float]
    affo_per_share: Optional[float]
    ffo_payout_ratio: Optional[float]
    affo_payout_ratio: Optional[float]


@dataclass
class NAVMetrics:
    """NAV analysis summary."""
    nav_estimate: Optional[float]
    nav_per_share: Optional[float]
    price_to_nav: Optional[float]
    premium_discount: Optional[float]


@dataclass
class REITValuation:
    """REIT valuation summary."""
    price_to_ffo: Optional[float]
    price_to_affo: Optional[float]
    implied_cap_rate: Optional[float]
    dividend_yield: Optional[float]
    assessment: str = ""


class REITAnalysis:
    """
    Comprehensive REIT analysis.

    Provides specialized analysis for Real Estate Investment Trusts.
    """

    def __init__(self, financials: REITFinancials):
        """Initialize with REIT financials."""
        self.financials = financials

    def analyze_ffo(self) -> FFOMetrics:
        """
        Analyze FFO and AFFO.

        Returns:
            FFO metrics including per-share and payout ratios
        """
        f = self.financials

        ffo = funds_from_operations(
            f.net_income,
            f.depreciation_real_estate,
            f.gains_on_property_sales,
            f.impairments
        )

        affo = adjusted_ffo(
            ffo,
            f.straight_line_rent_adjustment,
            f.recurring_capex,
            f.tenant_improvements,
            f.leasing_commissions,
            f.amortization_above_below_market_leases
        )

        ffo_ps = ffo_per_share(ffo, f.shares_outstanding)
        affo_ps = affo_per_share(affo, f.shares_outstanding)

        total_dividends = f.dividends_per_share * f.shares_outstanding

        return FFOMetrics(
            ffo=ffo,
            affo=affo,
            ffo_per_share=ffo_ps,
            affo_per_share=affo_ps,
            ffo_payout_ratio=dividend_payout_ffo(total_dividends, ffo),
            affo_payout_ratio=dividend_payout_affo(total_dividends, affo)
        )

    def analyze_nav(
        self,
        cap_rate: Optional[float] = None
    ) -> NAVMetrics:
        """
        Analyze Net Asset Value.

        Args:
            cap_rate: Cap rate for valuation (if not provided, uses implied)

        Returns:
            NAV metrics
        """
        f = self.financials

        # If cap rate provided, use it; otherwise estimate
        if cap_rate is None:
            # Use book value as rough NAV estimate
            nav = net_asset_value(
                f.real_estate_assets_net,
                f.total_assets - f.real_estate_assets_net,
                f.total_liabilities
            )
        else:
            noi = f.net_operating_income
            nav = nav_from_cap_rate(
                noi,
                cap_rate,
                f.total_assets - f.real_estate_assets_gross,
                f.total_liabilities
            )

        nav_ps = None
        p_nav = None
        premium = None

        if nav is not None:
            nav_ps = nav_per_share(nav, f.shares_outstanding)
            if nav_ps is not None and nav_ps > 0:
                p_nav = price_to_nav(f.stock_price, nav_ps)
                premium = (f.stock_price - nav_ps) / nav_ps

        return NAVMetrics(
            nav_estimate=nav,
            nav_per_share=nav_ps,
            price_to_nav=p_nav,
            premium_discount=premium
        )

    def analyze_valuation(self) -> REITValuation:
        """
        Analyze REIT valuation.

        Returns:
            Valuation metrics with assessment
        """
        f = self.financials

        ffo_metrics = self.analyze_ffo()

        p_ffo = price_to_ffo(f.stock_price, ffo_metrics.ffo_per_share)
        p_affo = price_to_affo(f.stock_price, ffo_metrics.affo_per_share)
        impl_cap = implied_cap_rate(f.net_operating_income, f.enterprise_value)
        div_yield = dividend_yield_reit(f.dividends_per_share, f.stock_price)

        # Assessment
        assessment_parts = []

        if p_ffo is not None:
            if p_ffo < 12:
                assessment_parts.append("Attractive P/FFO")
            elif p_ffo > 20:
                assessment_parts.append("Expensive P/FFO")

        if impl_cap is not None:
            if impl_cap > 0.07:
                assessment_parts.append("High implied cap rate")
            elif impl_cap < 0.04:
                assessment_parts.append("Low implied cap rate")

        if div_yield is not None:
            if div_yield > 0.05:
                assessment_parts.append("High dividend yield")
            elif div_yield < 0.03:
                assessment_parts.append("Low dividend yield")

        return REITValuation(
            price_to_ffo=p_ffo,
            price_to_affo=p_affo,
            implied_cap_rate=impl_cap,
            dividend_yield=div_yield,
            assessment=" | ".join(assessment_parts) if assessment_parts else "Moderate valuation"
        )

    def analyze_leverage(self) -> Dict[str, Optional[float]]:
        """
        Analyze debt and leverage metrics.

        Returns:
            Dictionary of leverage metrics
        """
        f = self.financials

        # Calculate EBITDA from NOI and G&A
        ebitda = f.net_operating_income - f.general_admin_expenses

        return {
            'debt_to_ebitda': debt_to_ebitda_reit(f.total_debt, ebitda),
            'debt_to_gross_assets': debt_to_gross_assets(
                f.total_debt,
                f.real_estate_assets_gross,
                f.total_assets - f.real_estate_assets_gross
            ),
            'fixed_charge_coverage': fixed_charge_coverage_reit(
                ebitda,
                f.interest_expense
            ),
            'interest_coverage': interest_coverage_reit(ebitda, f.interest_expense),
            'debt_to_market_cap': (
                f.total_debt / f.market_cap if f.market_cap > 0 else None
            ),
        }

    def analyze_operations(self) -> Dict[str, Optional[float]]:
        """
        Analyze property operations.

        Returns:
            Dictionary of operational metrics
        """
        f = self.financials

        return {
            'occupancy': f.occupancy_rate,
            'same_store_noi_growth': same_store_noi_growth(
                f.same_store_noi_current,
                f.same_store_noi_prior
            ),
            'noi_margin': noi_margin(
                f.net_operating_income,
                f.total_revenues
            ),
            'mark_to_market': mark_to_market(
                f.average_rent_per_sqft,
                f.market_rent_per_sqft
            ),
            'walt': f.weighted_average_lease_expiry,
            'revenue_per_sqft': (
                f.rental_revenue / f.total_square_feet
                if f.total_square_feet > 0 else None
            ),
        }

    def full_analysis(self) -> Dict:
        """
        Perform complete REIT analysis.

        Returns:
            Dictionary with all analysis components
        """
        return {
            'ffo_analysis': self.analyze_ffo(),
            'nav_analysis': self.analyze_nav(),
            'valuation': self.analyze_valuation(),
            'leverage': self.analyze_leverage(),
            'operations': self.analyze_operations(),
        }
