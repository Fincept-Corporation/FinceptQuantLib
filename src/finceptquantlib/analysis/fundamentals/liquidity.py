"""
Liquidity Ratios

Comprehensive liquidity analysis with 10+ ratios:
- Current Ratio, Quick Ratio, Cash Ratio
- Working Capital Ratios
- Cash Conversion Cycle
- Operating Cash Flow Coverage
- Defensive Interval

All ratios help assess a company's ability to meet short-term obligations.
"""

from dataclasses import dataclass
from typing import Optional, Dict
from .statements import (
    IncomeStatement, BalanceSheet, CashFlowStatement, FinancialStatements
)


@dataclass
class LiquidityResult:
    """Result of liquidity analysis."""
    # Basic Ratios
    current_ratio: float = 0.0
    quick_ratio: float = 0.0
    cash_ratio: float = 0.0

    # Working Capital
    working_capital: float = 0.0
    working_capital_ratio: float = 0.0
    net_working_capital_to_assets: float = 0.0

    # Cash Cycle
    days_inventory: float = 0.0
    days_receivables: float = 0.0
    days_payables: float = 0.0
    cash_conversion_cycle: float = 0.0
    operating_cycle: float = 0.0

    # Cash Flow Coverage
    operating_cash_flow_ratio: float = 0.0
    cash_coverage_ratio: float = 0.0
    defensive_interval: float = 0.0

    # Burn Rate
    cash_burn_rate: float = 0.0
    cash_runway_days: float = 0.0


# =============================================================================
# BASIC LIQUIDITY RATIOS
# =============================================================================

def current_ratio(balance: BalanceSheet) -> float:
    """
    Current Ratio = Current Assets / Current Liabilities

    Measures ability to pay short-term obligations with short-term assets.
    The most common liquidity measure.

    Benchmark: 1.5-2.0 is typically healthy, varies by industry
    - >2.0: Very liquid, may indicate inefficient asset use
    - 1.0-2.0: Adequate liquidity
    - <1.0: Potential liquidity problems
    """
    if balance.total_current_liabilities == 0:
        return float('inf') if balance.total_current_assets > 0 else 0.0
    return balance.total_current_assets / balance.total_current_liabilities


def quick_ratio(balance: BalanceSheet) -> float:
    """
    Quick Ratio = (Current Assets - Inventory) / Current Liabilities
    Also known as: Acid-Test Ratio

    More conservative than current ratio; excludes inventory which
    may not be easily converted to cash.

    Benchmark: >1.0 is typically healthy
    - >1.5: Strong liquidity position
    - 1.0-1.5: Adequate
    - <1.0: May struggle to meet short-term obligations
    """
    if balance.total_current_liabilities == 0:
        return float('inf') if balance.total_current_assets > balance.inventory else 0.0
    quick_assets = balance.total_current_assets - balance.inventory
    return quick_assets / balance.total_current_liabilities


def cash_ratio(balance: BalanceSheet) -> float:
    """
    Cash Ratio = (Cash + Short-term Investments) / Current Liabilities

    Most conservative liquidity measure; only considers the most
    liquid assets.

    Benchmark: 0.2-0.5 is typical
    - >0.5: Very conservative, may be inefficient
    - 0.2-0.5: Adequate
    - <0.2: Relying on other current assets for liquidity
    """
    if balance.total_current_liabilities == 0:
        return float('inf') if balance.cash_and_equivalents > 0 else 0.0
    cash_and_investments = balance.cash_and_equivalents + balance.short_term_investments
    return cash_and_investments / balance.total_current_liabilities


def absolute_liquidity_ratio(balance: BalanceSheet) -> float:
    """
    Absolute Liquidity Ratio = Cash / Current Liabilities

    Most stringent liquidity test; only uses cash.
    Also known as: Cash Liquidity Ratio, Super Quick Ratio

    Benchmark: 0.5-1.0 depending on industry
    """
    if balance.total_current_liabilities == 0:
        return float('inf') if balance.cash_and_equivalents > 0 else 0.0
    return balance.cash_and_equivalents / balance.total_current_liabilities


# =============================================================================
# WORKING CAPITAL RATIOS
# =============================================================================

def working_capital(balance: BalanceSheet) -> float:
    """
    Working Capital = Current Assets - Current Liabilities
    Also known as: Net Working Capital

    Absolute measure of short-term liquidity buffer.

    Positive: Company has buffer to meet short-term obligations
    Negative: Company may face liquidity challenges
    """
    return balance.working_capital


def working_capital_ratio(balance: BalanceSheet) -> float:
    """
    Working Capital Ratio = Working Capital / Total Assets

    Measures the proportion of assets tied up in working capital.

    Benchmark: 10-20% is typical for most industries
    """
    if balance.total_assets == 0:
        return 0.0
    return balance.working_capital / balance.total_assets


def working_capital_to_sales(
    income: IncomeStatement,
    balance: BalanceSheet
) -> float:
    """
    Working Capital to Sales = Working Capital / Revenue

    Measures working capital requirements relative to sales volume.

    Lower is better; indicates efficient working capital management.
    """
    if income.revenue == 0:
        return 0.0
    return balance.working_capital / income.revenue


def current_assets_to_total_assets(balance: BalanceSheet) -> float:
    """
    Current Assets to Total Assets = Current Assets / Total Assets

    Measures asset liquidity composition.
    """
    if balance.total_assets == 0:
        return 0.0
    return balance.total_current_assets / balance.total_assets


# =============================================================================
# CASH CONVERSION CYCLE
# =============================================================================

def days_inventory_outstanding(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Days Inventory Outstanding = (Avg Inventory / COGS) × 365
    Also known as: DIO, Days Inventory

    Measures how long inventory sits before being sold.

    Lower is generally better; indicates faster inventory turnover.

    Benchmark: Varies widely by industry
    - Retail: 30-60 days
    - Manufacturing: 60-120 days
    """
    if income.cost_of_goods_sold == 0:
        return 0.0

    if prior_balance:
        avg_inventory = (balance.inventory + prior_balance.inventory) / 2
    else:
        avg_inventory = balance.inventory

    return (avg_inventory / income.cost_of_goods_sold) * 365


def days_sales_outstanding(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Days Sales Outstanding = (Avg Receivables / Revenue) × 365
    Also known as: DSO, Days Receivables, Average Collection Period

    Measures how long it takes to collect receivables.

    Lower is better; indicates faster collection.

    Benchmark: 30-45 days is typical
    """
    if income.revenue == 0:
        return 0.0

    if prior_balance:
        avg_receivables = (balance.accounts_receivable + prior_balance.accounts_receivable) / 2
    else:
        avg_receivables = balance.accounts_receivable

    return (avg_receivables / income.revenue) * 365


def days_payables_outstanding(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Days Payables Outstanding = (Avg Payables / COGS) × 365
    Also known as: DPO, Days Payables

    Measures how long it takes to pay suppliers.

    Higher can be better (free financing from suppliers), but
    too high may strain supplier relationships.

    Benchmark: 30-60 days typical
    """
    if income.cost_of_goods_sold == 0:
        return 0.0

    if prior_balance:
        avg_payables = (balance.accounts_payable + prior_balance.accounts_payable) / 2
    else:
        avg_payables = balance.accounts_payable

    return (avg_payables / income.cost_of_goods_sold) * 365


def cash_conversion_cycle(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Cash Conversion Cycle = DIO + DSO - DPO

    Measures the time between paying for inventory and
    receiving cash from customers.

    Lower is better; indicates faster conversion of
    investments back to cash.

    Negative CCC: Company receives customer payment before paying suppliers
    (e.g., Amazon, Dell's old direct model)

    Benchmark: Varies by industry; improvement over time is positive
    """
    dio = days_inventory_outstanding(income, balance, prior_balance)
    dso = days_sales_outstanding(income, balance, prior_balance)
    dpo = days_payables_outstanding(income, balance, prior_balance)

    return dio + dso - dpo


def operating_cycle(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Operating Cycle = DIO + DSO

    Measures the time from acquiring inventory to collecting
    cash from sales.

    Lower is better; indicates more efficient operations.
    """
    dio = days_inventory_outstanding(income, balance, prior_balance)
    dso = days_sales_outstanding(income, balance, prior_balance)

    return dio + dso


# =============================================================================
# CASH FLOW COVERAGE RATIOS
# =============================================================================

def operating_cash_flow_ratio(
    cash_flow: CashFlowStatement,
    balance: BalanceSheet
) -> float:
    """
    Operating Cash Flow Ratio = Cash from Operations / Current Liabilities

    Cash-based alternative to current ratio.
    Measures ability to cover short-term obligations with operating cash flow.

    Benchmark: >1.0 indicates strong cash generation
    """
    if balance.total_current_liabilities == 0:
        return float('inf') if cash_flow.cash_from_operations > 0 else 0.0
    return cash_flow.cash_from_operations / balance.total_current_liabilities


def cash_coverage_ratio(
    income: IncomeStatement,
    cash_flow: CashFlowStatement
) -> float:
    """
    Cash Coverage Ratio = (CFO + Interest + Taxes) / Interest Expense

    Measures ability to cover interest with cash flow.
    More conservative than interest coverage ratio.

    Benchmark: >2.0 indicates comfortable coverage
    """
    if income.interest_expense == 0:
        return float('inf') if cash_flow.cash_from_operations > 0 else 0.0

    # Add back interest and taxes to CFO
    ebitda_proxy = cash_flow.cash_from_operations + income.interest_expense + income.income_tax
    return ebitda_proxy / income.interest_expense


def defensive_interval(
    income: IncomeStatement,
    balance: BalanceSheet
) -> float:
    """
    Defensive Interval = Liquid Assets / Daily Operating Expenses

    Measures how many days a company can operate using only
    liquid assets without additional revenue.

    Liquid Assets = Cash + Short-term Investments + Receivables
    Daily Expenses = (COGS + Operating Expenses - Depreciation) / 365

    Benchmark: Higher is better; 60-90 days provides good buffer
    """
    liquid_assets = (
        balance.cash_and_equivalents +
        balance.short_term_investments +
        balance.accounts_receivable
    )

    # Operating expenses excluding non-cash items
    daily_expenses = (
        income.cost_of_goods_sold +
        income.operating_expenses -
        income.depreciation -
        income.amortization
    ) / 365

    if daily_expenses <= 0:
        return float('inf') if liquid_assets > 0 else 0.0

    return liquid_assets / daily_expenses


def interval_measure(
    income: IncomeStatement,
    balance: BalanceSheet
) -> float:
    """
    Interval Measure = Quick Assets / Average Daily Expenditure

    Similar to defensive interval but uses quick assets.

    Benchmark: Higher indicates more days of coverage
    """
    quick_assets = (
        balance.total_current_assets - balance.inventory
    )

    daily_expenses = (
        income.cost_of_goods_sold +
        income.operating_expenses -
        income.depreciation -
        income.amortization
    ) / 365

    if daily_expenses <= 0:
        return float('inf') if quick_assets > 0 else 0.0

    return quick_assets / daily_expenses


# =============================================================================
# CASH BURN AND RUNWAY
# =============================================================================

def cash_burn_rate(cash_flow: CashFlowStatement) -> float:
    """
    Cash Burn Rate = -CFO (if negative) / 12

    Monthly rate at which a company is consuming cash.
    Relevant for startups and companies with negative operating cash flow.

    Returns: Monthly cash burn (positive number if burning cash)
    """
    if cash_flow.cash_from_operations >= 0:
        return 0.0  # Not burning cash
    return -cash_flow.cash_from_operations / 12


def cash_runway(
    balance: BalanceSheet,
    cash_flow: CashFlowStatement
) -> float:
    """
    Cash Runway = Cash / Monthly Burn Rate

    Months of operation remaining at current burn rate.
    Critical metric for startups and distressed companies.

    Returns: Months of runway (inf if not burning cash)
    """
    burn = cash_burn_rate(cash_flow)
    if burn <= 0:
        return float('inf')  # Not burning cash
    return balance.cash_and_equivalents / burn


def net_cash_position(balance: BalanceSheet) -> float:
    """
    Net Cash Position = Cash + Short-term Investments - Short-term Debt

    Shows available cash after covering short-term debt.

    Positive: Company has excess liquid resources
    Negative: Short-term debt exceeds liquid assets
    """
    return (
        balance.cash_and_equivalents +
        balance.short_term_investments -
        balance.short_term_debt
    )


# =============================================================================
# COMPREHENSIVE ANALYSIS CLASS
# =============================================================================

class LiquidityAnalysis:
    """
    Comprehensive liquidity analysis.

    Calculates all liquidity ratios for a company.

    Example:
        analysis = LiquidityAnalysis(statements, prior_statements)
        result = analysis.analyze()
        print(f"Current Ratio: {result.current_ratio:.2f}")
        print(f"Cash Conversion Cycle: {result.cash_conversion_cycle:.0f} days")
    """

    def __init__(
        self,
        current: FinancialStatements,
        prior: Optional[FinancialStatements] = None
    ):
        self.current = current
        self.prior = prior

    def analyze(self) -> LiquidityResult:
        """Calculate all liquidity ratios."""
        income = self.current.income_statement
        balance = self.current.balance_sheet
        cash_flow_stmt = self.current.cash_flow
        prior_balance = self.prior.balance_sheet if self.prior else None

        # Calculate cash cycle components
        dio = days_inventory_outstanding(income, balance, prior_balance)
        dso = days_sales_outstanding(income, balance, prior_balance)
        dpo = days_payables_outstanding(income, balance, prior_balance)

        burn = cash_burn_rate(cash_flow_stmt)

        return LiquidityResult(
            # Basic Ratios
            current_ratio=current_ratio(balance),
            quick_ratio=quick_ratio(balance),
            cash_ratio=cash_ratio(balance),

            # Working Capital
            working_capital=working_capital(balance),
            working_capital_ratio=working_capital_ratio(balance),
            net_working_capital_to_assets=working_capital_ratio(balance),

            # Cash Cycle
            days_inventory=dio,
            days_receivables=dso,
            days_payables=dpo,
            cash_conversion_cycle=dio + dso - dpo,
            operating_cycle=dio + dso,

            # Cash Flow Coverage
            operating_cash_flow_ratio=operating_cash_flow_ratio(cash_flow_stmt, balance),
            cash_coverage_ratio=cash_coverage_ratio(income, cash_flow_stmt),
            defensive_interval=defensive_interval(income, balance),

            # Burn Rate
            cash_burn_rate=burn,
            cash_runway_days=cash_runway(balance, cash_flow_stmt) * 30 if burn > 0 else float('inf'),
        )

    def to_dict(self) -> Dict[str, float]:
        """Return all ratios as dictionary."""
        result = self.analyze()
        return {
            'current_ratio': result.current_ratio,
            'quick_ratio': result.quick_ratio,
            'cash_ratio': result.cash_ratio,
            'working_capital': result.working_capital,
            'working_capital_ratio': result.working_capital_ratio,
            'days_inventory': result.days_inventory,
            'days_receivables': result.days_receivables,
            'days_payables': result.days_payables,
            'cash_conversion_cycle': result.cash_conversion_cycle,
            'operating_cycle': result.operating_cycle,
            'operating_cash_flow_ratio': result.operating_cash_flow_ratio,
            'cash_coverage_ratio': result.cash_coverage_ratio,
            'defensive_interval': result.defensive_interval,
        }

    def assess_liquidity_health(self) -> str:
        """Provide a qualitative assessment of liquidity health."""
        result = self.analyze()

        issues = []
        strengths = []

        # Current Ratio Assessment
        if result.current_ratio >= 2.0:
            strengths.append("Strong current ratio indicates ample short-term coverage")
        elif result.current_ratio >= 1.5:
            strengths.append("Adequate current ratio")
        elif result.current_ratio >= 1.0:
            issues.append("Current ratio on the low end; monitor closely")
        else:
            issues.append("Current ratio below 1.0; potential liquidity risk")

        # Quick Ratio Assessment
        if result.quick_ratio >= 1.0:
            strengths.append("Quick ratio indicates good immediate liquidity")
        else:
            issues.append("Quick ratio below 1.0; relies heavily on inventory")

        # Cash Conversion Cycle Assessment
        if result.cash_conversion_cycle < 0:
            strengths.append("Negative CCC; excellent working capital management")
        elif result.cash_conversion_cycle <= 60:
            strengths.append("Efficient cash conversion cycle")
        elif result.cash_conversion_cycle <= 120:
            pass  # Neutral
        else:
            issues.append("Long cash conversion cycle; working capital may be tied up")

        # Defensive Interval Assessment
        if result.defensive_interval >= 90:
            strengths.append("Strong defensive interval; can sustain operations for 90+ days")
        elif result.defensive_interval >= 60:
            strengths.append("Adequate defensive interval")
        elif result.defensive_interval >= 30:
            issues.append("Limited defensive interval; less than 60 days coverage")
        else:
            issues.append("Very low defensive interval; liquidity concerns")

        assessment = "LIQUIDITY ASSESSMENT\n" + "=" * 40 + "\n\n"

        if strengths:
            assessment += "STRENGTHS:\n"
            for s in strengths:
                assessment += f"  + {s}\n"
            assessment += "\n"

        if issues:
            assessment += "CONCERNS:\n"
            for i in issues:
                assessment += f"  - {i}\n"

        if not issues:
            assessment += "\nOVERALL: Strong liquidity position\n"
        elif len(issues) <= 1:
            assessment += "\nOVERALL: Adequate liquidity with minor concerns\n"
        else:
            assessment += "\nOVERALL: Liquidity requires attention\n"

        return assessment


# Alias for backward compatibility
days_inventory = days_inventory_outstanding
days_receivables = days_sales_outstanding
days_payables = days_payables_outstanding
