"""
Earnings Quality Analysis

Comprehensive earnings quality assessment:
- Accruals Analysis (Sloan, Balance Sheet, Cash Flow)
- Earnings Persistence
- Revenue Quality
- Cash Conversion Quality
- Red Flags Detection

High-quality earnings are sustainable, backed by cash, and not manipulated.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
from .statements import (
    IncomeStatement, BalanceSheet, CashFlowStatement, FinancialStatements
)


@dataclass
class EarningsQualityResult:
    """Result of earnings quality analysis."""
    # Accruals Measures
    sloan_accruals_ratio: float = 0.0
    balance_sheet_accruals: float = 0.0
    cash_flow_accruals: float = 0.0
    total_accruals: float = 0.0

    # Quality Ratios
    cash_conversion_quality: float = 0.0
    earnings_persistence_proxy: float = 0.0
    revenue_quality_proxy: float = 0.0

    # Warning Signals
    accruals_warning: bool = False
    cash_quality_warning: bool = False
    overall_quality_score: float = 0.0  # 0-100 scale


# =============================================================================
# ACCRUALS ANALYSIS
# =============================================================================

def sloan_accruals_ratio(
    income: IncomeStatement,
    cash_flow: CashFlowStatement,
    balance: BalanceSheet
) -> float:
    """
    Sloan Accruals Ratio = (Net Income - CFO) / Average Total Assets

    Richard Sloan's (1996) accruals measure.
    Higher accruals associated with lower earnings quality and lower future returns.

    Benchmark:
    - Near 0: High quality (earnings = cash)
    - Positive high: Accrual-heavy earnings (concern)
    - Negative: Cash exceeds earnings (conservative)

    Research threshold: >10% is a warning sign
    """
    if balance.total_assets == 0:
        return 0.0

    accruals = income.net_income - cash_flow.cash_from_operations
    return accruals / balance.total_assets


def balance_sheet_accruals(
    balance: BalanceSheet,
    prior_balance: BalanceSheet
) -> float:
    """
    Balance Sheet Accruals = Change in NOA / Average Total Assets

    NOA (Net Operating Assets) = Operating Assets - Operating Liabilities

    Measures accruals from balance sheet perspective.

    Higher values indicate more aggressive accounting.
    """
    # Calculate NOA (approximation)
    # Operating Assets = Total Assets - Cash - Short-term Investments
    # Operating Liabilities = Total Liabilities - Total Debt

    op_assets_current = (
        balance.total_assets -
        balance.cash_and_equivalents -
        balance.short_term_investments
    )
    op_liab_current = balance.total_liabilities - balance.total_debt
    noa_current = op_assets_current - op_liab_current

    op_assets_prior = (
        prior_balance.total_assets -
        prior_balance.cash_and_equivalents -
        prior_balance.short_term_investments
    )
    op_liab_prior = prior_balance.total_liabilities - prior_balance.total_debt
    noa_prior = op_assets_prior - op_liab_prior

    delta_noa = noa_current - noa_prior
    avg_assets = (balance.total_assets + prior_balance.total_assets) / 2

    if avg_assets == 0:
        return 0.0
    return delta_noa / avg_assets


def cash_flow_accruals(
    income: IncomeStatement,
    cash_flow: CashFlowStatement
) -> float:
    """
    Cash Flow Accruals = Net Income - Operating Cash Flow

    Simple measure of the gap between earnings and cash.

    Negative = Good (cash > earnings)
    Positive = Concern (earnings > cash)
    """
    return income.net_income - cash_flow.cash_from_operations


def total_accruals_to_assets(
    income: IncomeStatement,
    cash_flow: CashFlowStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Total Accruals to Assets = Total Accruals / Average Total Assets

    Total Accruals = Net Income - CFO (from cash flow) + D&A
    Or = Change in NOA (from balance sheet)

    Normalized measure of total accruals.
    """
    if prior_balance:
        avg_assets = (balance.total_assets + prior_balance.total_assets) / 2
    else:
        avg_assets = balance.total_assets

    if avg_assets == 0:
        return 0.0

    total_accruals = income.net_income - cash_flow.cash_from_operations
    return total_accruals / avg_assets


# =============================================================================
# CASH CONVERSION QUALITY
# =============================================================================

def cash_conversion_quality(
    cash_flow: CashFlowStatement,
    income: IncomeStatement
) -> float:
    """
    Cash Conversion Quality = Operating Cash Flow / Net Income

    Measures how much of earnings is converted to cash.

    Benchmark:
    - >1.0: Excellent; cash exceeds reported earnings
    - 0.8-1.0: Good
    - 0.5-0.8: Moderate; some accruals concern
    - <0.5: Poor; significant accruals
    """
    if income.net_income == 0:
        return float('inf') if cash_flow.cash_from_operations > 0 else 0.0
    return cash_flow.cash_from_operations / income.net_income


def cash_conversion_to_ebitda(
    cash_flow: CashFlowStatement,
    income: IncomeStatement
) -> float:
    """
    Cash Conversion = Operating Cash Flow / EBITDA

    Alternative cash conversion measure.

    Benchmark: Should be close to 1.0 for quality earnings
    """
    if income.ebitda == 0:
        return float('inf') if cash_flow.cash_from_operations > 0 else 0.0
    return cash_flow.cash_from_operations / income.ebitda


def operating_cash_flow_to_ebit(
    cash_flow: CashFlowStatement,
    income: IncomeStatement
) -> float:
    """
    CFO to EBIT = Operating Cash Flow / EBIT

    Measures cash generation relative to operating profit.
    """
    if income.operating_income == 0:
        return float('inf') if cash_flow.cash_from_operations > 0 else 0.0
    return cash_flow.cash_from_operations / income.operating_income


# =============================================================================
# EARNINGS PERSISTENCE
# =============================================================================

def earnings_persistence(
    current_income: IncomeStatement,
    prior_income: IncomeStatement
) -> float:
    """
    Earnings Persistence Proxy = Current ROA vs Prior ROA correlation

    Higher persistence indicates more predictable, higher-quality earnings.
    This is a simplified proxy; true persistence requires multi-year regression.

    Returns ratio of current to prior earnings (simplified persistence indicator)
    """
    if prior_income.net_income == 0:
        return 0.0

    # Simple ratio as persistence proxy
    return current_income.net_income / prior_income.net_income


def earnings_variability(
    incomes: List[IncomeStatement]
) -> float:
    """
    Earnings Variability = Standard Deviation of Earnings / Mean Earnings

    Coefficient of variation of earnings.
    Lower variability suggests more stable, predictable earnings.

    Args:
        incomes: List of income statements over multiple periods

    Returns: Coefficient of variation (lower is better)
    """
    if len(incomes) < 2:
        return 0.0

    earnings = [inc.net_income for inc in incomes]
    mean_earnings = sum(earnings) / len(earnings)

    if mean_earnings == 0:
        return float('inf')

    variance = sum((e - mean_earnings) ** 2 for e in earnings) / len(earnings)
    std_dev = variance ** 0.5

    return std_dev / abs(mean_earnings)


def earnings_smoothness(
    incomes: List[IncomeStatement],
    cash_flows: List[CashFlowStatement]
) -> float:
    """
    Earnings Smoothness = Std(Net Income) / Std(CFO)

    Smooth earnings relative to volatile cash flows suggests manipulation.

    Benchmark:
    - Near 1.0: Normal
    - <<1.0: Possibly smoothed earnings
    """
    if len(incomes) < 3 or len(cash_flows) < 3:
        return 1.0

    ni_values = [inc.net_income for inc in incomes]
    cfo_values = [cf.cash_from_operations for cf in cash_flows]

    mean_ni = sum(ni_values) / len(ni_values)
    mean_cfo = sum(cfo_values) / len(cfo_values)

    var_ni = sum((ni - mean_ni) ** 2 for ni in ni_values) / len(ni_values)
    var_cfo = sum((cfo - mean_cfo) ** 2 for cfo in cfo_values) / len(cfo_values)

    std_ni = var_ni ** 0.5
    std_cfo = var_cfo ** 0.5

    if std_cfo == 0:
        return 1.0

    return std_ni / std_cfo


# =============================================================================
# REVENUE QUALITY
# =============================================================================

def receivables_to_revenue_change(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_income: IncomeStatement,
    prior_balance: BalanceSheet
) -> float:
    """
    Change in (Receivables / Revenue)

    Rising receivables relative to revenue may indicate:
    - Channel stuffing
    - Aggressive revenue recognition
    - Deteriorating collection

    Benchmark:
    - Negative or ~0: Good
    - Significantly positive: Warning sign
    """
    if income.revenue == 0 or prior_income.revenue == 0:
        return 0.0

    current_ratio = balance.accounts_receivable / income.revenue
    prior_ratio = prior_balance.accounts_receivable / prior_income.revenue

    return current_ratio - prior_ratio


def days_sales_in_receivables_change(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_income: IncomeStatement,
    prior_balance: BalanceSheet
) -> float:
    """
    Change in Days Sales in Receivables

    DSR = (AR / Revenue) × 365

    Significant increase is a warning sign.
    """
    if income.revenue == 0 or prior_income.revenue == 0:
        return 0.0

    current_dsr = (balance.accounts_receivable / income.revenue) * 365
    prior_dsr = (prior_balance.accounts_receivable / prior_income.revenue) * 365

    return current_dsr - prior_dsr


def deferred_revenue_change(
    balance: BalanceSheet,
    prior_balance: BalanceSheet,
    income: IncomeStatement
) -> float:
    """
    Deferred Revenue Change / Revenue

    Declining deferred revenue may indicate future revenue weakness.
    (Note: Requires deferred revenue in balance sheet data)

    This is a placeholder - actual implementation would require
    deferred revenue data in the balance sheet.
    """
    # Placeholder - would need deferred_revenue field
    return 0.0


# =============================================================================
# RED FLAGS AND WARNING SIGNALS
# =============================================================================

def detect_earnings_red_flags(
    income: IncomeStatement,
    balance: BalanceSheet,
    cash_flow: CashFlowStatement,
    prior_income: Optional[IncomeStatement] = None,
    prior_balance: Optional[BalanceSheet] = None
) -> Dict[str, bool]:
    """
    Detect common earnings quality red flags.

    Returns dictionary of warning signals.
    """
    flags = {}

    # 1. Large gap between earnings and cash flow
    accruals = income.net_income - cash_flow.cash_from_operations
    if balance.total_assets > 0:
        accrual_ratio = accruals / balance.total_assets
        flags['high_accruals'] = accrual_ratio > 0.10
    else:
        flags['high_accruals'] = False

    # 2. Earnings growing faster than cash flow
    if prior_income and cash_flow.cash_from_operations != 0:
        # Compare to CFO growth (would need prior CFO)
        flags['earnings_outpacing_cash'] = income.net_income > cash_flow.cash_from_operations * 1.5

    # 3. Receivables growing faster than revenue
    if prior_income and prior_balance:
        rev_growth = (income.revenue - prior_income.revenue) / prior_income.revenue if prior_income.revenue != 0 else 0
        ar_growth = (balance.accounts_receivable - prior_balance.accounts_receivable) / prior_balance.accounts_receivable if prior_balance.accounts_receivable != 0 else 0
        flags['receivables_outpacing_revenue'] = ar_growth > rev_growth + 0.05

    # 4. Inventory growing faster than COGS
    if prior_income and prior_balance:
        cogs_growth = (income.cost_of_goods_sold - prior_income.cost_of_goods_sold) / prior_income.cost_of_goods_sold if prior_income.cost_of_goods_sold != 0 else 0
        inv_growth = (balance.inventory - prior_balance.inventory) / prior_balance.inventory if prior_balance.inventory != 0 else 0
        flags['inventory_buildup'] = inv_growth > cogs_growth + 0.10

    # 5. Soft assets (intangibles, goodwill) growing
    if prior_balance:
        soft_assets = balance.goodwill + balance.intangible_assets
        prior_soft = prior_balance.goodwill + prior_balance.intangible_assets
        soft_growth = (soft_assets - prior_soft) / prior_soft if prior_soft != 0 else 0
        flags['soft_assets_growing'] = soft_growth > 0.20
    else:
        flags['soft_assets_growing'] = False

    # 6. Aggressive revenue recognition signal
    flags['aggressive_revenue'] = income.net_income > cash_flow.cash_from_operations * 1.3 if cash_flow.cash_from_operations > 0 else False

    return flags


def earnings_quality_score(
    income: IncomeStatement,
    balance: BalanceSheet,
    cash_flow: CashFlowStatement,
    prior_income: Optional[IncomeStatement] = None,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Calculate overall earnings quality score (0-100).

    Higher score = higher quality earnings.

    Components:
    - Cash conversion (30 points)
    - Low accruals (30 points)
    - No red flags (40 points)
    """
    score = 0.0

    # Cash conversion component (30 points)
    ccq = cash_conversion_quality(cash_flow, income)
    if ccq >= 1.0:
        score += 30
    elif ccq >= 0.8:
        score += 25
    elif ccq >= 0.5:
        score += 15
    else:
        score += 5

    # Accruals component (30 points)
    accrual_ratio = sloan_accruals_ratio(income, cash_flow, balance)
    if accrual_ratio <= 0:
        score += 30  # Negative accruals = very good
    elif accrual_ratio <= 0.05:
        score += 25
    elif accrual_ratio <= 0.10:
        score += 15
    else:
        score += 5

    # Red flags component (40 points)
    flags = detect_earnings_red_flags(income, balance, cash_flow, prior_income, prior_balance)
    flag_count = sum(1 for f in flags.values() if f)

    if flag_count == 0:
        score += 40
    elif flag_count == 1:
        score += 30
    elif flag_count == 2:
        score += 20
    elif flag_count == 3:
        score += 10
    else:
        score += 0

    return score


# =============================================================================
# COMPREHENSIVE ANALYSIS CLASS
# =============================================================================

class EarningsQualityAnalysis:
    """
    Comprehensive earnings quality analysis.

    Calculates all earnings quality metrics.

    Example:
        analysis = EarningsQualityAnalysis(statements, prior_statements)
        result = analysis.analyze()
        print(f"Sloan Accruals: {result.sloan_accruals_ratio:.2%}")
        print(f"Quality Score: {result.overall_quality_score:.0f}/100")
    """

    def __init__(
        self,
        current: FinancialStatements,
        prior: Optional[FinancialStatements] = None
    ):
        self.current = current
        self.prior = prior

    def analyze(self) -> EarningsQualityResult:
        """Calculate all earnings quality metrics."""
        income = self.current.income_statement
        balance = self.current.balance_sheet
        cash_flow_stmt = self.current.cash_flow

        prior_income = self.prior.income_statement if self.prior else None
        prior_balance = self.prior.balance_sheet if self.prior else None

        # Accruals
        sloan = sloan_accruals_ratio(income, cash_flow_stmt, balance)
        cf_accruals = cash_flow_accruals(income, cash_flow_stmt)

        if prior_balance:
            bs_accruals = balance_sheet_accruals(balance, prior_balance)
        else:
            bs_accruals = 0.0

        # Quality ratios
        ccq = cash_conversion_quality(cash_flow_stmt, income)

        # Persistence proxy
        if prior_income and prior_income.net_income != 0:
            persistence = earnings_persistence(income, prior_income)
        else:
            persistence = 1.0

        # Quality score
        quality_score = earnings_quality_score(
            income, balance, cash_flow_stmt, prior_income, prior_balance
        )

        # Warning flags
        accruals_warning = sloan > 0.10
        cash_warning = ccq < 0.5 if isinstance(ccq, float) and ccq != float('inf') else False

        return EarningsQualityResult(
            sloan_accruals_ratio=sloan,
            balance_sheet_accruals=bs_accruals,
            cash_flow_accruals=cf_accruals,
            total_accruals=total_accruals_to_assets(income, cash_flow_stmt, balance, prior_balance),
            cash_conversion_quality=ccq,
            earnings_persistence_proxy=persistence,
            revenue_quality_proxy=1.0,  # Placeholder
            accruals_warning=accruals_warning,
            cash_quality_warning=cash_warning,
            overall_quality_score=quality_score,
        )

    def to_dict(self) -> Dict[str, float]:
        """Return all metrics as dictionary."""
        result = self.analyze()
        return {
            'sloan_accruals_ratio': result.sloan_accruals_ratio,
            'balance_sheet_accruals': result.balance_sheet_accruals,
            'cash_flow_accruals': result.cash_flow_accruals,
            'cash_conversion_quality': result.cash_conversion_quality,
            'earnings_persistence_proxy': result.earnings_persistence_proxy,
            'overall_quality_score': result.overall_quality_score,
        }

    def get_red_flags(self) -> Dict[str, bool]:
        """Get red flag indicators."""
        income = self.current.income_statement
        balance = self.current.balance_sheet
        cash_flow_stmt = self.current.cash_flow
        prior_income = self.prior.income_statement if self.prior else None
        prior_balance = self.prior.balance_sheet if self.prior else None

        return detect_earnings_red_flags(
            income, balance, cash_flow_stmt, prior_income, prior_balance
        )

    def assess_quality(self) -> str:
        """Provide qualitative assessment of earnings quality."""
        result = self.analyze()
        red_flags = self.get_red_flags()

        assessment = "EARNINGS QUALITY ASSESSMENT\n" + "=" * 40 + "\n\n"
        assessment += f"Overall Quality Score: {result.overall_quality_score:.0f}/100\n\n"

        # Rating
        if result.overall_quality_score >= 80:
            rating = "HIGH QUALITY"
        elif result.overall_quality_score >= 60:
            rating = "GOOD QUALITY"
        elif result.overall_quality_score >= 40:
            rating = "MODERATE QUALITY"
        else:
            rating = "LOW QUALITY"

        assessment += f"Rating: {rating}\n\n"

        # Key Metrics
        assessment += "KEY METRICS:\n"
        assessment += f"  Sloan Accruals Ratio: {result.sloan_accruals_ratio:.2%}\n"
        ccq_display = f"{result.cash_conversion_quality:.2f}" if result.cash_conversion_quality != float('inf') else "N/A"
        assessment += f"  Cash Conversion Quality: {ccq_display}\n\n"

        # Red Flags
        active_flags = [k for k, v in red_flags.items() if v]
        if active_flags:
            assessment += "WARNING FLAGS:\n"
            for flag in active_flags:
                flag_name = flag.replace('_', ' ').title()
                assessment += f"  ! {flag_name}\n"
        else:
            assessment += "No warning flags detected.\n"

        # Interpretation
        assessment += "\nINTERPRETATION:\n"
        if result.sloan_accruals_ratio < 0:
            assessment += "  + Negative accruals indicate conservative accounting\n"
        elif result.sloan_accruals_ratio > 0.10:
            assessment += "  - High accruals suggest aggressive accounting\n"

        if isinstance(result.cash_conversion_quality, float) and result.cash_conversion_quality >= 1.0:
            assessment += "  + Cash flow exceeds reported earnings\n"
        elif isinstance(result.cash_conversion_quality, float) and result.cash_conversion_quality < 0.8:
            assessment += "  - Earnings not fully backed by cash flow\n"

        return assessment
