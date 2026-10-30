"""
Cash Flow Analysis

Comprehensive cash flow analysis with 15+ metrics:
- Free Cash Flow (FCF, FCFF, FCFE)
- Cash Flow Margins
- Cash Flow Coverage Ratios
- Cash Flow Quality Metrics
- Cash Generation Analysis

These metrics help assess the quality and sustainability of cash generation.
"""

from dataclasses import dataclass
from typing import Optional, Dict
from .statements import (
    IncomeStatement, BalanceSheet, CashFlowStatement, FinancialStatements
)


@dataclass
class CashFlowResult:
    """Result of cash flow analysis."""
    # Free Cash Flow Metrics
    free_cash_flow: float = 0.0
    fcff: float = 0.0
    fcfe: float = 0.0
    levered_fcf: float = 0.0
    unlevered_fcf: float = 0.0

    # Cash Flow Margins
    operating_cash_flow_margin: float = 0.0
    free_cash_flow_margin: float = 0.0
    cash_flow_to_net_income: float = 0.0

    # Coverage Ratios
    cash_coverage_of_capex: float = 0.0
    cash_coverage_of_dividends: float = 0.0
    cash_coverage_of_debt_service: float = 0.0

    # Quality Metrics
    cash_flow_quality: float = 0.0
    accruals_ratio: float = 0.0
    operating_cash_flow_ratio: float = 0.0

    # Growth & Reinvestment
    reinvestment_rate: float = 0.0
    cash_return_on_assets: float = 0.0


# =============================================================================
# FREE CASH FLOW CALCULATIONS
# =============================================================================

def free_cash_flow(cash_flow: CashFlowStatement) -> float:
    """
    Free Cash Flow = Operating Cash Flow - Capital Expenditures

    The simplest and most common FCF definition.
    Represents cash available for distribution to investors.

    Benchmark: Positive FCF is generally positive; growing FCF over time is ideal
    """
    return cash_flow.cash_from_operations - cash_flow.capex


def fcff(
    income: IncomeStatement,
    balance: BalanceSheet,
    cash_flow: CashFlowStatement,
    prior_balance: Optional[BalanceSheet] = None,
    tax_rate: Optional[float] = None
) -> float:
    """
    Free Cash Flow to Firm (FCFF)

    FCFF = EBIT × (1 - Tax Rate) + D&A - CapEx - ΔWorking Capital

    Cash available to ALL capital providers (debt + equity).
    Used in WACC-based DCF valuation.

    Args:
        income: Income statement
        balance: Current balance sheet
        cash_flow: Cash flow statement
        prior_balance: Prior period balance sheet (for working capital change)
        tax_rate: Tax rate to use (defaults to effective tax rate)

    Returns: FCFF
    """
    if tax_rate is None:
        tax_rate = income.effective_tax_rate

    # NOPAT = EBIT × (1 - Tax Rate)
    nopat = income.operating_income * (1 - tax_rate)

    # Depreciation & Amortization
    da = income.depreciation + income.amortization

    # Change in Working Capital
    if prior_balance:
        delta_wc = balance.working_capital - prior_balance.working_capital
    else:
        # Use change from cash flow statement if available
        delta_wc = -(
            cash_flow.change_in_receivables +
            cash_flow.change_in_inventory -
            cash_flow.change_in_payables
        )

    return nopat + da - cash_flow.capex - delta_wc


def fcfe(
    income: IncomeStatement,
    balance: BalanceSheet,
    cash_flow: CashFlowStatement,
    prior_balance: Optional[BalanceSheet] = None,
    tax_rate: Optional[float] = None
) -> float:
    """
    Free Cash Flow to Equity (FCFE)

    FCFE = FCFF - Interest × (1 - Tax Rate) + Net Borrowing

    Cash available to equity holders after all obligations.
    Used in cost-of-equity based DCF valuation.

    Args:
        income: Income statement
        balance: Current balance sheet
        cash_flow: Cash flow statement
        prior_balance: Prior period balance sheet
        tax_rate: Tax rate to use

    Returns: FCFE
    """
    if tax_rate is None:
        tax_rate = income.effective_tax_rate

    # Start with FCFF
    fcff_value = fcff(income, balance, cash_flow, prior_balance, tax_rate)

    # Subtract after-tax interest expense
    after_tax_interest = income.interest_expense * (1 - tax_rate)

    # Add net borrowing (new debt issued minus debt repaid)
    net_borrowing = cash_flow.debt_issued - cash_flow.debt_repaid

    return fcff_value - after_tax_interest + net_borrowing


def unlevered_free_cash_flow(
    income: IncomeStatement,
    balance: BalanceSheet,
    cash_flow: CashFlowStatement,
    prior_balance: Optional[BalanceSheet] = None,
    tax_rate: Optional[float] = None
) -> float:
    """
    Unlevered Free Cash Flow = FCFF

    Cash flow before financing effects. Same as FCFF.
    Used for enterprise value calculations.
    """
    return fcff(income, balance, cash_flow, prior_balance, tax_rate)


def levered_free_cash_flow(
    cash_flow: CashFlowStatement,
    income: IncomeStatement
) -> float:
    """
    Levered Free Cash Flow = CFO - CapEx - Mandatory Debt Payments

    A practical measure of cash available for equity after debt service.
    """
    # Using mandatory debt service (interest)
    return cash_flow.cash_from_operations - cash_flow.capex - income.interest_expense


# =============================================================================
# CASH FLOW MARGINS
# =============================================================================

def operating_cash_flow_margin(
    cash_flow: CashFlowStatement,
    income: IncomeStatement
) -> float:
    """
    Operating Cash Flow Margin = CFO / Revenue

    Cash-based alternative to operating margin.

    Benchmark:
    - >15%: Strong cash generation
    - 10-15%: Good
    - 5-10%: Moderate
    - <5%: Weak cash conversion
    """
    if income.revenue == 0:
        return 0.0
    return cash_flow.cash_from_operations / income.revenue


def free_cash_flow_margin(
    cash_flow: CashFlowStatement,
    income: IncomeStatement
) -> float:
    """
    Free Cash Flow Margin = FCF / Revenue

    Measures FCF generation per dollar of revenue.

    Benchmark:
    - >10%: Excellent
    - 5-10%: Good
    - 0-5%: Moderate
    - Negative: Concern
    """
    if income.revenue == 0:
        return 0.0
    fcf = free_cash_flow(cash_flow)
    return fcf / income.revenue


def cash_flow_to_net_income(
    cash_flow: CashFlowStatement,
    income: IncomeStatement
) -> float:
    """
    Cash Flow to Net Income Ratio = CFO / Net Income
    Also known as: Quality of Earnings

    Measures how much of net income is backed by cash.

    Benchmark:
    - >1.0: Good; cash exceeds reported earnings
    - ~1.0: Normal
    - <1.0: Concern; earnings exceed cash generation
    - Significantly <1.0: Potential earnings quality issues
    """
    if income.net_income == 0:
        return float('inf') if cash_flow.cash_from_operations > 0 else 0.0
    return cash_flow.cash_from_operations / income.net_income


def cash_earnings_ratio(
    cash_flow: CashFlowStatement,
    income: IncomeStatement
) -> float:
    """
    Cash Earnings = CFO / Net Income

    Alias for cash_flow_to_net_income. Higher is better.
    """
    return cash_flow_to_net_income(cash_flow, income)


# =============================================================================
# CASH FLOW COVERAGE RATIOS
# =============================================================================

def cash_coverage_of_capex(cash_flow: CashFlowStatement) -> float:
    """
    Cash Coverage of CapEx = CFO / Capital Expenditures

    Measures ability to fund capital investment from operations.

    Benchmark:
    - >2.0x: Strong; can fund growth and distribute cash
    - 1.0-2.0x: Adequate; can fund maintenance capex
    - <1.0x: Needs external financing for capex
    """
    if cash_flow.capex == 0:
        return float('inf') if cash_flow.cash_from_operations > 0 else 0.0
    return cash_flow.cash_from_operations / cash_flow.capex


def cash_coverage_of_dividends(cash_flow: CashFlowStatement) -> float:
    """
    Cash Coverage of Dividends = CFO / Dividends Paid

    Measures ability to sustain dividend payments.

    Benchmark:
    - >2.0x: Dividend well covered
    - 1.5-2.0x: Comfortable
    - 1.0-1.5x: Tight coverage
    - <1.0x: Dividend at risk; borrowing to pay dividends
    """
    if cash_flow.dividends_paid == 0:
        return float('inf') if cash_flow.cash_from_operations > 0 else 0.0
    return cash_flow.cash_from_operations / cash_flow.dividends_paid


def cash_coverage_of_debt_service(
    cash_flow: CashFlowStatement,
    income: IncomeStatement,
    principal_payments: float = 0.0
) -> float:
    """
    Cash Coverage of Debt Service = CFO / (Interest + Principal)

    Measures ability to service all debt obligations.

    Args:
        cash_flow: Cash flow statement
        income: Income statement (for interest)
        principal_payments: Annual principal payments due

    Benchmark:
    - >1.5x: Comfortable
    - 1.0-1.5x: Tight
    - <1.0x: Cannot cover debt service from operations
    """
    debt_service = income.interest_expense + principal_payments
    if debt_service == 0:
        return float('inf') if cash_flow.cash_from_operations > 0 else 0.0
    return cash_flow.cash_from_operations / debt_service


def cash_coverage_of_fixed_charges(
    cash_flow: CashFlowStatement,
    income: IncomeStatement,
    lease_payments: float = 0.0
) -> float:
    """
    Cash Coverage of Fixed Charges = CFO / (Interest + Leases)

    Cash-based fixed charge coverage.
    """
    fixed_charges = income.interest_expense + lease_payments
    if fixed_charges == 0:
        return float('inf') if cash_flow.cash_from_operations > 0 else 0.0
    return cash_flow.cash_from_operations / fixed_charges


# =============================================================================
# CASH FLOW QUALITY METRICS
# =============================================================================

def quality_of_earnings(
    cash_flow: CashFlowStatement,
    income: IncomeStatement
) -> float:
    """
    Quality of Earnings = CFO / Net Income

    High-quality earnings are backed by cash flow.
    Low quality earnings have high accruals.

    Benchmark:
    - >1.0: High quality; cash exceeds earnings
    - 0.8-1.0: Normal
    - <0.8: Lower quality; high accrual component
    """
    return cash_flow_to_net_income(cash_flow, income)


def accruals_ratio(
    income: IncomeStatement,
    cash_flow: CashFlowStatement,
    balance: BalanceSheet
) -> float:
    """
    Accruals Ratio = (Net Income - CFO) / Total Assets

    Measures the accrual component of earnings.
    Higher accruals associated with lower earnings quality.

    Benchmark:
    - Near 0: High quality (cash = earnings)
    - Positive: Accrual-based earnings
    - Negative: Cash exceeds earnings (conservative)

    Research shows high accruals predict lower future returns.
    """
    if balance.total_assets == 0:
        return 0.0
    accruals = income.net_income - cash_flow.cash_from_operations
    return accruals / balance.total_assets


def cash_flow_accruals(
    income: IncomeStatement,
    cash_flow: CashFlowStatement
) -> float:
    """
    Cash Flow Accruals = Net Income - CFO

    Absolute measure of accruals in earnings.

    Negative: Cash exceeds earnings (good)
    Positive: Earnings exceed cash (accruals present)
    """
    return income.net_income - cash_flow.cash_from_operations


# =============================================================================
# REINVESTMENT AND GROWTH METRICS
# =============================================================================

def reinvestment_rate(
    income: IncomeStatement,
    cash_flow: CashFlowStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None,
    tax_rate: Optional[float] = None
) -> float:
    """
    Reinvestment Rate = Net Investment / NOPAT

    Net Investment = CapEx - Depreciation + ΔWorking Capital

    Measures proportion of operating profit reinvested.

    Benchmark:
    - High rate + High ROIC = Growth (good)
    - High rate + Low ROIC = Value destruction
    - Low rate = Mature company returning cash
    """
    if tax_rate is None:
        tax_rate = income.effective_tax_rate

    nopat = income.operating_income * (1 - tax_rate)
    if nopat <= 0:
        return 0.0

    # Net investment
    da = income.depreciation + income.amortization
    net_capex = cash_flow.capex - da

    if prior_balance:
        delta_wc = balance.working_capital - prior_balance.working_capital
    else:
        delta_wc = 0

    net_investment = net_capex + delta_wc

    return net_investment / nopat


def cash_return_on_assets(
    cash_flow: CashFlowStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Cash Return on Assets = CFO / Average Total Assets

    Cash-based ROA. More reliable than accrual-based ROA.

    Benchmark: Compare to accrual ROA; should be similar or higher
    """
    if prior_balance:
        avg_assets = (balance.total_assets + prior_balance.total_assets) / 2
    else:
        avg_assets = balance.total_assets

    if avg_assets == 0:
        return 0.0
    return cash_flow.cash_from_operations / avg_assets


def cash_return_on_equity(
    cash_flow: CashFlowStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Cash Return on Equity = CFO / Average Total Equity

    Cash-based ROE.
    """
    if prior_balance:
        avg_equity = (balance.total_equity + prior_balance.total_equity) / 2
    else:
        avg_equity = balance.total_equity

    if avg_equity == 0:
        return 0.0
    return cash_flow.cash_from_operations / avg_equity


# =============================================================================
# CASH FLOW DECOMPOSITION
# =============================================================================

def cash_flow_composition(cash_flow: CashFlowStatement) -> Dict[str, float]:
    """
    Analyze the composition of total cash flows.

    Returns breakdown by activity type.
    """
    total_change = cash_flow.net_change_in_cash

    return {
        'operating_pct': cash_flow.cash_from_operations / total_change if total_change != 0 else 0.0,
        'investing_pct': cash_flow.cash_from_investing / total_change if total_change != 0 else 0.0,
        'financing_pct': cash_flow.cash_from_financing / total_change if total_change != 0 else 0.0,
        'operating': cash_flow.cash_from_operations,
        'investing': cash_flow.cash_from_investing,
        'financing': cash_flow.cash_from_financing,
        'net_change': total_change,
    }


def cash_flow_sustainability(
    cash_flow: CashFlowStatement,
    income: IncomeStatement
) -> Dict[str, bool]:
    """
    Assess sustainability of cash flows.

    Returns flags indicating sustainability concerns.
    """
    fcf = free_cash_flow(cash_flow)

    return {
        'positive_cfo': cash_flow.cash_from_operations > 0,
        'positive_fcf': fcf > 0,
        'cfo_covers_capex': cash_flow.cash_from_operations > cash_flow.capex,
        'cfo_covers_dividends': cash_flow.cash_from_operations > cash_flow.dividends_paid,
        'quality_earnings': cash_flow.cash_from_operations >= income.net_income * 0.8,
        'not_debt_funded': cash_flow.debt_issued < cash_flow.capex + cash_flow.dividends_paid,
    }


# =============================================================================
# COMPREHENSIVE ANALYSIS CLASS
# =============================================================================

class CashFlowAnalysis:
    """
    Comprehensive cash flow analysis.

    Calculates all cash flow metrics for a company.

    Example:
        analysis = CashFlowAnalysis(statements, prior_statements)
        result = analysis.analyze()
        print(f"FCF: ${result.free_cash_flow/1e6:.1f}M")
        print(f"FCF Margin: {result.free_cash_flow_margin:.1%}")
    """

    def __init__(
        self,
        current: FinancialStatements,
        prior: Optional[FinancialStatements] = None
    ):
        self.current = current
        self.prior = prior

    def analyze(self) -> CashFlowResult:
        """Calculate all cash flow metrics."""
        income = self.current.income_statement
        balance = self.current.balance_sheet
        cash_flow_stmt = self.current.cash_flow
        prior_balance = self.prior.balance_sheet if self.prior else None

        fcf = free_cash_flow(cash_flow_stmt)
        fcff_value = fcff(income, balance, cash_flow_stmt, prior_balance)
        fcfe_value = fcfe(income, balance, cash_flow_stmt, prior_balance)

        return CashFlowResult(
            # Free Cash Flow Metrics
            free_cash_flow=fcf,
            fcff=fcff_value,
            fcfe=fcfe_value,
            levered_fcf=levered_free_cash_flow(cash_flow_stmt, income),
            unlevered_fcf=fcff_value,

            # Cash Flow Margins
            operating_cash_flow_margin=operating_cash_flow_margin(cash_flow_stmt, income),
            free_cash_flow_margin=free_cash_flow_margin(cash_flow_stmt, income),
            cash_flow_to_net_income=cash_flow_to_net_income(cash_flow_stmt, income),

            # Coverage Ratios
            cash_coverage_of_capex=cash_coverage_of_capex(cash_flow_stmt),
            cash_coverage_of_dividends=cash_coverage_of_dividends(cash_flow_stmt),
            cash_coverage_of_debt_service=cash_coverage_of_debt_service(
                cash_flow_stmt, income
            ),

            # Quality Metrics
            cash_flow_quality=quality_of_earnings(cash_flow_stmt, income),
            accruals_ratio=accruals_ratio(income, cash_flow_stmt, balance),
            operating_cash_flow_ratio=cash_flow_stmt.cash_from_operations / balance.total_current_liabilities if balance.total_current_liabilities > 0 else 0.0,

            # Growth & Reinvestment
            reinvestment_rate=reinvestment_rate(income, cash_flow_stmt, balance, prior_balance),
            cash_return_on_assets=cash_return_on_assets(cash_flow_stmt, balance, prior_balance),
        )

    def to_dict(self) -> Dict[str, float]:
        """Return all metrics as dictionary."""
        result = self.analyze()
        return {
            'free_cash_flow': result.free_cash_flow,
            'fcff': result.fcff,
            'fcfe': result.fcfe,
            'operating_cash_flow_margin': result.operating_cash_flow_margin,
            'free_cash_flow_margin': result.free_cash_flow_margin,
            'cash_flow_to_net_income': result.cash_flow_to_net_income,
            'cash_coverage_of_capex': result.cash_coverage_of_capex,
            'cash_coverage_of_dividends': result.cash_coverage_of_dividends,
            'cash_flow_quality': result.cash_flow_quality,
            'accruals_ratio': result.accruals_ratio,
            'reinvestment_rate': result.reinvestment_rate,
        }

    def assess_cash_flow_health(self) -> str:
        """Provide a qualitative assessment of cash flow health."""
        result = self.analyze()
        sustainability = cash_flow_sustainability(self.current.cash_flow, self.current.income_statement)

        strengths = []
        concerns = []

        # Positive FCF
        if result.free_cash_flow > 0:
            strengths.append("Generating positive free cash flow")
        else:
            concerns.append("Negative free cash flow")

        # FCF Margin
        if result.free_cash_flow_margin > 0.10:
            strengths.append("Strong FCF margin (>10%)")
        elif result.free_cash_flow_margin < 0.05 and result.free_cash_flow_margin > 0:
            concerns.append("Low FCF margin (<5%)")

        # Earnings Quality
        if result.cash_flow_quality >= 1.0:
            strengths.append("High earnings quality; cash exceeds net income")
        elif result.cash_flow_quality < 0.8 and result.cash_flow_quality > 0:
            concerns.append("Lower earnings quality; significant accruals")

        # Accruals
        if result.accruals_ratio < 0:
            strengths.append("Negative accruals (conservative accounting)")
        elif result.accruals_ratio > 0.10:
            concerns.append("High accruals ratio (>10%)")

        # CapEx Coverage
        if result.cash_coverage_of_capex > 2.0:
            strengths.append("Strong CFO coverage of CapEx")
        elif result.cash_coverage_of_capex < 1.0 and result.cash_coverage_of_capex > 0:
            concerns.append("CFO doesn't cover CapEx")

        # Dividend Coverage
        if sustainability['cfo_covers_dividends']:
            strengths.append("Dividends covered by operating cash flow")
        elif self.current.cash_flow.dividends_paid > 0:
            concerns.append("Dividends exceed operating cash flow")

        assessment = "CASH FLOW HEALTH ASSESSMENT\n" + "=" * 40 + "\n\n"

        if strengths:
            assessment += "STRENGTHS:\n"
            for s in strengths:
                assessment += f"  + {s}\n"
            assessment += "\n"

        if concerns:
            assessment += "CONCERNS:\n"
            for c in concerns:
                assessment += f"  - {c}\n"

        score = len(strengths) - len(concerns)
        if score >= 2:
            assessment += "\nOVERALL: STRONG cash flow generation\n"
        elif score >= 0:
            assessment += "\nOVERALL: ADEQUATE cash flow\n"
        else:
            assessment += "\nOVERALL: Cash flow CONCERNS warrant attention\n"

        return assessment


# Convenience function aliases
cash_flow_margin = operating_cash_flow_margin
