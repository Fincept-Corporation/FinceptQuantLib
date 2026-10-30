"""
Growth Analysis

Comprehensive growth analysis:
- Sustainable Growth Rate
- Internal Growth Rate
- PRAT Model
- Compound Annual Growth Rate (CAGR)
- Growth Decomposition
- Implied Growth from ROIC

These metrics help analyze and forecast company growth.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from .statements import (
    IncomeStatement, BalanceSheet, CashFlowStatement, FinancialStatements
)


@dataclass
class GrowthResult:
    """Result of growth analysis."""
    # Sustainable Growth
    sustainable_growth_rate: float = 0.0
    internal_growth_rate: float = 0.0
    prat_growth: float = 0.0
    implied_growth_roic: float = 0.0

    # Historical Growth Rates
    revenue_growth: float = 0.0
    earnings_growth: float = 0.0
    ebitda_growth: float = 0.0
    fcf_growth: float = 0.0
    asset_growth: float = 0.0
    equity_growth: float = 0.0

    # Per-Share Growth
    eps_growth: float = 0.0
    bvps_growth: float = 0.0
    dps_growth: float = 0.0

    # Quality Metrics
    earnings_quality_of_growth: float = 0.0
    organic_growth: float = 0.0


# =============================================================================
# SUSTAINABLE GROWTH MODELS
# =============================================================================

def sustainable_growth_rate(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Sustainable Growth Rate = ROE × Retention Rate

    The rate at which a company can grow using internally generated funds
    while maintaining constant debt/equity ratio.

    Retention Rate = 1 - Dividend Payout Ratio = (NI - Dividends) / NI

    Benchmark: Compare to actual growth; if actual > SGR, company needs external financing.
    """
    # Calculate ROE
    if prior_balance:
        avg_equity = (balance.total_equity + prior_balance.total_equity) / 2
    else:
        avg_equity = balance.total_equity

    if avg_equity == 0:
        return 0.0

    roe = income.net_income / avg_equity

    # Calculate retention rate
    if income.net_income == 0:
        retention = 0.0
    else:
        dividends_paid = income.dividends_per_share * income.shares_outstanding
        retention = (income.net_income - dividends_paid) / income.net_income
        retention = max(0, min(1, retention))  # Bound between 0 and 1

    return roe * retention


def internal_growth_rate(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Internal Growth Rate = (ROA × b) / (1 - ROA × b)

    Where b = retention rate

    The maximum growth rate achievable without any external financing
    (neither debt nor equity).

    More conservative than SGR.
    """
    # Calculate ROA
    if prior_balance:
        avg_assets = (balance.total_assets + prior_balance.total_assets) / 2
    else:
        avg_assets = balance.total_assets

    if avg_assets == 0:
        return 0.0

    roa = income.net_income / avg_assets

    # Calculate retention rate
    if income.net_income == 0:
        return 0.0

    dividends_paid = income.dividends_per_share * income.shares_outstanding
    retention = (income.net_income - dividends_paid) / income.net_income
    retention = max(0, min(1, retention))

    # IGR formula
    roa_b = roa * retention
    if roa_b >= 1:
        return float('inf')  # Mathematically undefined
    if roa_b <= -1:
        return 0.0

    return roa_b / (1 - roa_b)


def prat_model(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> Tuple[float, Dict[str, float]]:
    """
    PRAT Model: Sustainable Growth Rate Decomposition

    g = P × R × A × T

    Where:
    - P = Profit Margin (Net Income / Revenue)
    - R = Retention Rate (1 - Payout Ratio)
    - A = Asset Turnover (Revenue / Assets)
    - T = Financial Leverage (Assets / Equity)

    Returns both growth rate and component breakdown.
    """
    # Calculate averages
    if prior_balance:
        avg_assets = (balance.total_assets + prior_balance.total_assets) / 2
        avg_equity = (balance.total_equity + prior_balance.total_equity) / 2
    else:
        avg_assets = balance.total_assets
        avg_equity = balance.total_equity

    # P: Profit Margin
    if income.revenue == 0:
        p = 0.0
    else:
        p = income.net_income / income.revenue

    # R: Retention Rate
    if income.net_income == 0:
        r = 0.0
    else:
        dividends = income.dividends_per_share * income.shares_outstanding
        r = (income.net_income - dividends) / income.net_income
        r = max(0, min(1, r))

    # A: Asset Turnover
    if avg_assets == 0:
        a = 0.0
    else:
        a = income.revenue / avg_assets

    # T: Financial Leverage
    if avg_equity == 0:
        t = float('inf') if avg_assets > 0 else 0.0
    else:
        t = avg_assets / avg_equity

    growth_rate = p * r * a * t

    components = {
        'profit_margin': p,
        'retention_rate': r,
        'asset_turnover': a,
        'financial_leverage': t,
        'growth_rate': growth_rate
    }

    return growth_rate, components


def implied_growth_from_roic(
    income: IncomeStatement,
    balance: BalanceSheet,
    cash_flow: CashFlowStatement,
    prior_balance: Optional[BalanceSheet] = None,
    tax_rate: Optional[float] = None
) -> float:
    """
    Implied Growth Rate = ROIC × Reinvestment Rate

    Growth implied by the company's return on capital and reinvestment.
    Fundamental value creation equation.

    Reinvestment Rate = (Net CapEx + ΔWC) / NOPAT
    """
    if tax_rate is None:
        tax_rate = income.effective_tax_rate

    # Calculate NOPAT
    nopat = income.operating_income * (1 - tax_rate)
    if nopat <= 0:
        return 0.0

    # Calculate ROIC
    if prior_balance:
        avg_ic = (balance.invested_capital + prior_balance.invested_capital) / 2
    else:
        avg_ic = balance.invested_capital

    if avg_ic == 0:
        return 0.0

    roic = nopat / avg_ic

    # Calculate Reinvestment Rate
    da = income.depreciation + income.amortization
    net_capex = cash_flow.capex - da

    if prior_balance:
        delta_wc = balance.working_capital - prior_balance.working_capital
    else:
        delta_wc = 0

    reinvestment = net_capex + delta_wc
    reinvestment_rate = reinvestment / nopat if nopat > 0 else 0.0

    return roic * reinvestment_rate


# =============================================================================
# HISTORICAL GROWTH CALCULATIONS
# =============================================================================

def year_over_year_growth(
    current: float,
    prior: float
) -> float:
    """
    Year-over-Year Growth = (Current - Prior) / |Prior|

    Returns growth rate as decimal.
    """
    if prior == 0:
        return float('inf') if current > 0 else 0.0
    return (current - prior) / abs(prior)


def cagr(
    values: List[float],
    years: Optional[int] = None
) -> float:
    """
    Compound Annual Growth Rate

    CAGR = (Ending Value / Beginning Value)^(1/n) - 1

    Args:
        values: List of values (first to last)
        years: Number of years (defaults to len(values) - 1)

    Returns: CAGR as decimal
    """
    if len(values) < 2:
        return 0.0

    beginning = values[0]
    ending = values[-1]

    if years is None:
        years = len(values) - 1

    if years <= 0 or beginning <= 0 or ending <= 0:
        return 0.0

    return (ending / beginning) ** (1 / years) - 1


def revenue_growth(
    current: IncomeStatement,
    prior: IncomeStatement
) -> float:
    """Revenue YoY Growth"""
    return year_over_year_growth(current.revenue, prior.revenue)


def earnings_growth(
    current: IncomeStatement,
    prior: IncomeStatement
) -> float:
    """Net Income YoY Growth"""
    return year_over_year_growth(current.net_income, prior.net_income)


def ebitda_growth(
    current: IncomeStatement,
    prior: IncomeStatement
) -> float:
    """EBITDA YoY Growth"""
    return year_over_year_growth(current.ebitda, prior.ebitda)


def eps_growth(
    current: IncomeStatement,
    prior: IncomeStatement
) -> float:
    """EPS YoY Growth"""
    return year_over_year_growth(current.eps_basic, prior.eps_basic)


def asset_growth(
    current: BalanceSheet,
    prior: BalanceSheet
) -> float:
    """Total Assets YoY Growth"""
    return year_over_year_growth(current.total_assets, prior.total_assets)


def equity_growth(
    current: BalanceSheet,
    prior: BalanceSheet
) -> float:
    """Shareholders' Equity YoY Growth"""
    return year_over_year_growth(current.total_equity, prior.total_equity)


def book_value_per_share_growth(
    current_balance: BalanceSheet,
    prior_balance: BalanceSheet,
    current_shares: float,
    prior_shares: float
) -> float:
    """Book Value Per Share YoY Growth"""
    if prior_shares == 0 or current_shares == 0:
        return 0.0

    current_bvps = current_balance.total_equity / current_shares
    prior_bvps = prior_balance.total_equity / prior_shares

    return year_over_year_growth(current_bvps, prior_bvps)


def fcf_growth(
    current: CashFlowStatement,
    prior: CashFlowStatement
) -> float:
    """Free Cash Flow YoY Growth"""
    current_fcf = current.cash_from_operations - current.capex
    prior_fcf = prior.cash_from_operations - prior.capex
    return year_over_year_growth(current_fcf, prior_fcf)


# =============================================================================
# GROWTH QUALITY ANALYSIS
# =============================================================================

def growth_quality_score(
    revenue_g: float,
    earnings_g: float,
    fcf_g: float
) -> float:
    """
    Assess quality of growth.

    High quality growth: Revenue, earnings, and FCF all growing in tandem.
    Low quality: Earnings growing faster than revenue (margin expansion only)
                 or FCF not keeping pace with earnings.

    Returns score 0-100.
    """
    score = 50.0  # Base score

    # Bonus for revenue growth
    if revenue_g > 0:
        score += min(20, revenue_g * 100)

    # Bonus for earnings keeping pace with revenue
    if revenue_g > 0 and earnings_g >= revenue_g * 0.8:
        score += 15
    elif earnings_g > revenue_g * 1.5:
        score -= 10  # Penalty for unsustainable margin expansion

    # Bonus for FCF keeping pace
    if earnings_g > 0 and fcf_g >= earnings_g * 0.7:
        score += 15
    elif fcf_g < 0 and earnings_g > 0:
        score -= 20  # Major penalty for FCF declining while earnings grow

    return max(0, min(100, score))


def organic_vs_acquisition_growth(
    current: BalanceSheet,
    prior: BalanceSheet,
    current_revenue: float,
    prior_revenue: float
) -> Dict[str, float]:
    """
    Estimate organic vs. acquisition-driven growth.

    Uses goodwill changes as proxy for acquisitions.

    Returns breakdown of growth sources.
    """
    total_revenue_growth = year_over_year_growth(current_revenue, prior_revenue)
    goodwill_growth = year_over_year_growth(current.goodwill, prior.goodwill)

    # Rough heuristic: significant goodwill increase suggests acquisitions
    if goodwill_growth > 0.1:  # More than 10% goodwill increase
        # Assume some portion of revenue growth is from acquisitions
        estimated_acquisition_growth = min(goodwill_growth * 0.5, total_revenue_growth)
        estimated_organic_growth = total_revenue_growth - estimated_acquisition_growth
    else:
        estimated_organic_growth = total_revenue_growth
        estimated_acquisition_growth = 0.0

    return {
        'total_growth': total_revenue_growth,
        'estimated_organic': estimated_organic_growth,
        'estimated_acquisition': estimated_acquisition_growth,
        'goodwill_change': goodwill_growth,
    }


# =============================================================================
# COMPREHENSIVE ANALYSIS CLASS
# =============================================================================

class GrowthAnalysis:
    """
    Comprehensive growth analysis.

    Calculates all growth metrics for a company.

    Example:
        analysis = GrowthAnalysis(statements, prior_statements)
        result = analysis.analyze()
        print(f"Sustainable Growth Rate: {result.sustainable_growth_rate:.1%}")
        print(f"Revenue Growth: {result.revenue_growth:.1%}")
    """

    def __init__(
        self,
        current: FinancialStatements,
        prior: Optional[FinancialStatements] = None
    ):
        self.current = current
        self.prior = prior

    def analyze(self) -> GrowthResult:
        """Calculate all growth metrics."""
        income = self.current.income_statement
        balance = self.current.balance_sheet
        cash_flow_stmt = self.current.cash_flow

        prior_income = self.prior.income_statement if self.prior else None
        prior_balance = self.prior.balance_sheet if self.prior else None
        prior_cf = self.prior.cash_flow if self.prior else None

        # Sustainable growth metrics
        sgr = sustainable_growth_rate(income, balance, prior_balance)
        igr = internal_growth_rate(income, balance, prior_balance)
        prat_g, _ = prat_model(income, balance, prior_balance)
        implied_g = implied_growth_from_roic(income, balance, cash_flow_stmt, prior_balance)

        # Historical growth (YoY)
        if prior_income:
            rev_g = revenue_growth(income, prior_income)
            earn_g = earnings_growth(income, prior_income)
            ebitda_g = ebitda_growth(income, prior_income)
            eps_g = eps_growth(income, prior_income)
        else:
            rev_g = earn_g = ebitda_g = eps_g = 0.0

        if prior_balance:
            asset_g = asset_growth(balance, prior_balance)
            equity_g = equity_growth(balance, prior_balance)
            bvps_g = book_value_per_share_growth(
                balance, prior_balance,
                income.shares_outstanding,
                prior_income.shares_outstanding if prior_income else income.shares_outstanding
            )
        else:
            asset_g = equity_g = bvps_g = 0.0

        if prior_cf:
            fcf_g = fcf_growth(cash_flow_stmt, prior_cf)
        else:
            fcf_g = 0.0

        # Quality metrics
        quality = growth_quality_score(rev_g, earn_g, fcf_g)

        # Organic growth estimate
        if prior_balance and prior_income:
            organic_data = organic_vs_acquisition_growth(
                balance, prior_balance, income.revenue, prior_income.revenue
            )
            organic_g = organic_data['estimated_organic']
        else:
            organic_g = rev_g

        return GrowthResult(
            sustainable_growth_rate=sgr,
            internal_growth_rate=igr,
            prat_growth=prat_g,
            implied_growth_roic=implied_g,
            revenue_growth=rev_g,
            earnings_growth=earn_g,
            ebitda_growth=ebitda_g,
            fcf_growth=fcf_g,
            asset_growth=asset_g,
            equity_growth=equity_g,
            eps_growth=eps_g,
            bvps_growth=bvps_g,
            dps_growth=0.0,  # Would need prior DPS
            earnings_quality_of_growth=quality,
            organic_growth=organic_g,
        )

    def to_dict(self) -> Dict[str, float]:
        """Return all metrics as dictionary."""
        result = self.analyze()
        return {
            'sustainable_growth_rate': result.sustainable_growth_rate,
            'internal_growth_rate': result.internal_growth_rate,
            'prat_growth': result.prat_growth,
            'implied_growth_roic': result.implied_growth_roic,
            'revenue_growth': result.revenue_growth,
            'earnings_growth': result.earnings_growth,
            'ebitda_growth': result.ebitda_growth,
            'fcf_growth': result.fcf_growth,
            'eps_growth': result.eps_growth,
            'earnings_quality_of_growth': result.earnings_quality_of_growth,
        }

    def prat_breakdown(self) -> Dict[str, float]:
        """Get PRAT model component breakdown."""
        prior_balance = self.prior.balance_sheet if self.prior else None
        _, components = prat_model(
            self.current.income_statement,
            self.current.balance_sheet,
            prior_balance
        )
        return components

    def assess_growth_sustainability(self) -> str:
        """Provide qualitative assessment of growth sustainability."""
        result = self.analyze()

        assessment = "GROWTH SUSTAINABILITY ASSESSMENT\n" + "=" * 40 + "\n\n"

        # Key rates
        assessment += f"Sustainable Growth Rate: {result.sustainable_growth_rate:.1%}\n"
        assessment += f"Internal Growth Rate: {result.internal_growth_rate:.1%}\n"
        assessment += f"Implied Growth (ROIC): {result.implied_growth_roic:.1%}\n"
        assessment += f"Actual Revenue Growth: {result.revenue_growth:.1%}\n\n"

        strengths = []
        concerns = []

        # Compare actual to sustainable
        if result.revenue_growth > 0 and result.sustainable_growth_rate > 0:
            if result.revenue_growth <= result.sustainable_growth_rate:
                strengths.append("Revenue growth within sustainable limits")
            elif result.revenue_growth <= result.sustainable_growth_rate * 1.2:
                concerns.append("Growth slightly above sustainable rate")
            else:
                concerns.append("Growth significantly above sustainable rate; may need external financing")

        # Quality of growth
        if result.earnings_quality_of_growth >= 70:
            strengths.append("High-quality balanced growth")
        elif result.earnings_quality_of_growth >= 50:
            pass  # Average
        else:
            concerns.append("Growth quality concerns; earnings/FCF diverging from revenue")

        # FCF growth
        if result.fcf_growth >= result.earnings_growth * 0.8:
            strengths.append("FCF growth tracking earnings growth")
        elif result.fcf_growth < 0 and result.earnings_growth > 0:
            concerns.append("FCF declining while earnings grow")

        if strengths:
            assessment += "STRENGTHS:\n"
            for s in strengths:
                assessment += f"  + {s}\n"
            assessment += "\n"

        if concerns:
            assessment += "CONCERNS:\n"
            for c in concerns:
                assessment += f"  - {c}\n"

        if len(concerns) == 0:
            assessment += "\nOVERALL: Growth appears SUSTAINABLE\n"
        elif len(concerns) <= 1:
            assessment += "\nOVERALL: Growth is MODERATELY sustainable\n"
        else:
            assessment += "\nOVERALL: Growth sustainability is QUESTIONABLE\n"

        return assessment
