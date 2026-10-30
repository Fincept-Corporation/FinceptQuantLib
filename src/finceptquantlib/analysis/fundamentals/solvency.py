"""
Solvency Ratios

Comprehensive solvency analysis with 15+ ratios:
- Debt Ratios (Debt/Equity, Debt/Assets, Debt/Capital)
- Coverage Ratios (Interest, Fixed Charge, Debt Service)
- Leverage Ratios (Financial Leverage, Equity Multiplier)
- Capital Structure Analysis

All ratios help assess a company's ability to meet long-term obligations.
"""

from dataclasses import dataclass
from typing import Optional, Dict
from .statements import (
    IncomeStatement, BalanceSheet, CashFlowStatement, FinancialStatements
)


@dataclass
class SolvencyResult:
    """Result of solvency analysis."""
    # Debt Ratios
    debt_to_equity: float = 0.0
    debt_to_assets: float = 0.0
    debt_to_capital: float = 0.0
    long_term_debt_to_equity: float = 0.0
    long_term_debt_to_capital: float = 0.0
    net_debt_to_equity: float = 0.0
    net_debt_to_ebitda: float = 0.0

    # Coverage Ratios
    interest_coverage: float = 0.0
    cash_interest_coverage: float = 0.0
    fixed_charge_coverage: float = 0.0
    debt_service_coverage: float = 0.0
    cash_flow_to_debt: float = 0.0

    # Leverage Ratios
    financial_leverage: float = 0.0
    equity_multiplier: float = 0.0
    capitalization_ratio: float = 0.0
    degree_of_financial_leverage: float = 0.0


# =============================================================================
# DEBT RATIOS
# =============================================================================

def debt_to_equity(balance: BalanceSheet) -> float:
    """
    Debt to Equity = Total Debt / Total Equity

    Measures the proportion of debt financing relative to equity.
    Key indicator of financial leverage.

    Benchmark:
    - 0.5-1.0: Conservative
    - 1.0-2.0: Moderate leverage
    - >2.0: High leverage

    Note: Varies significantly by industry (utilities/REITs typically higher)
    """
    if balance.total_equity == 0:
        return float('inf') if balance.total_debt > 0 else 0.0
    return balance.total_debt / balance.total_equity


def debt_to_assets(balance: BalanceSheet) -> float:
    """
    Debt to Assets = Total Debt / Total Assets

    Measures the percentage of assets financed by debt.

    Benchmark:
    - <30%: Conservative, low financial risk
    - 30-50%: Moderate leverage
    - >50%: High leverage, increased financial risk
    """
    if balance.total_assets == 0:
        return 0.0
    return balance.total_debt / balance.total_assets


def debt_to_capital(balance: BalanceSheet) -> float:
    """
    Debt to Capital = Total Debt / (Total Debt + Total Equity)

    Measures debt as a proportion of total capital employed.

    Benchmark:
    - <40%: Conservative
    - 40-60%: Moderate
    - >60%: Aggressive capital structure
    """
    total_capital = balance.total_debt + balance.total_equity
    if total_capital == 0:
        return 0.0
    return balance.total_debt / total_capital


def long_term_debt_to_equity(balance: BalanceSheet) -> float:
    """
    Long-term Debt to Equity = Long-term Debt / Total Equity

    Focuses on permanent debt financing.
    More relevant for capital structure analysis than total D/E.

    Benchmark: Similar to D/E but typically lower
    """
    if balance.total_equity == 0:
        return float('inf') if balance.long_term_debt > 0 else 0.0
    return balance.long_term_debt / balance.total_equity


def long_term_debt_to_capital(balance: BalanceSheet) -> float:
    """
    Long-term Debt to Capital = LT Debt / (LT Debt + Equity)

    Measures long-term debt as a proportion of permanent capital.

    Useful for analyzing capital structure decisions.
    """
    lt_capital = balance.long_term_debt + balance.total_equity
    if lt_capital == 0:
        return 0.0
    return balance.long_term_debt / lt_capital


def net_debt_to_equity(balance: BalanceSheet) -> float:
    """
    Net Debt to Equity = (Total Debt - Cash) / Total Equity

    Accounts for cash that could be used to repay debt.
    More meaningful for companies with large cash balances.

    Benchmark: Lower than gross D/E; can be negative if cash > debt
    """
    if balance.total_equity == 0:
        return float('inf') if balance.net_debt > 0 else 0.0
    return balance.net_debt / balance.total_equity


def net_debt_to_ebitda(
    income: IncomeStatement,
    balance: BalanceSheet
) -> float:
    """
    Net Debt to EBITDA = Net Debt / EBITDA

    Measures years of EBITDA needed to repay debt.
    Key covenant metric for lenders.

    Benchmark:
    - <1.5x: Very conservative
    - 1.5-3.0x: Healthy
    - 3.0-4.0x: Elevated but manageable
    - >4.0x: High leverage, may face financing constraints
    """
    if income.ebitda == 0:
        return float('inf') if balance.net_debt > 0 else 0.0
    return balance.net_debt / income.ebitda


def total_liabilities_to_assets(balance: BalanceSheet) -> float:
    """
    Total Liabilities to Assets = Total Liabilities / Total Assets

    Broader measure including all liabilities, not just debt.

    Benchmark: <60% is generally conservative
    """
    if balance.total_assets == 0:
        return 0.0
    return balance.total_liabilities / balance.total_assets


def liabilities_to_equity(balance: BalanceSheet) -> float:
    """
    Liabilities to Equity = Total Liabilities / Total Equity

    Broader than D/E; includes operating liabilities.

    Useful for analyzing total financial obligations.
    """
    if balance.total_equity == 0:
        return float('inf') if balance.total_liabilities > 0 else 0.0
    return balance.total_liabilities / balance.total_equity


# =============================================================================
# COVERAGE RATIOS
# =============================================================================

def interest_coverage(income: IncomeStatement) -> float:
    """
    Interest Coverage Ratio = EBIT / Interest Expense
    Also known as: Times Interest Earned (TIE)

    Measures ability to pay interest from operating earnings.

    Benchmark:
    - >5.0x: Very strong coverage
    - 3.0-5.0x: Comfortable
    - 2.0-3.0x: Adequate
    - <2.0x: Potential distress
    - <1.0x: Cannot cover interest with operations
    """
    if income.interest_expense == 0:
        return float('inf') if income.operating_income > 0 else 0.0
    return income.operating_income / income.interest_expense


def ebitda_interest_coverage(income: IncomeStatement) -> float:
    """
    EBITDA Interest Coverage = EBITDA / Interest Expense

    More conservative than EBIT coverage; accounts for capital expenditure
    requirements.

    Benchmark: Typically 1-2x higher than interest coverage
    """
    if income.interest_expense == 0:
        return float('inf') if income.ebitda > 0 else 0.0
    return income.ebitda / income.interest_expense


def cash_interest_coverage(
    cash_flow: CashFlowStatement,
    income: IncomeStatement
) -> float:
    """
    Cash Interest Coverage = (CFO + Interest + Taxes) / Interest Expense

    Cash-based interest coverage; more reliable than accrual-based.

    Benchmark: >3.0x indicates strong cash generation for interest
    """
    if income.interest_expense == 0:
        return float('inf') if cash_flow.cash_from_operations > 0 else 0.0

    # Add back interest and taxes to get cash before these items
    cash_for_interest = (
        cash_flow.cash_from_operations +
        income.interest_expense +
        income.income_tax
    )
    return cash_for_interest / income.interest_expense


def fixed_charge_coverage(
    income: IncomeStatement,
    lease_payments: float = 0.0
) -> float:
    """
    Fixed Charge Coverage = (EBIT + Lease Payments) / (Interest + Lease Payments)

    Expands interest coverage to include lease obligations.
    Required by many loan covenants.

    Args:
        income: Income statement
        lease_payments: Annual lease payments (operating leases)

    Benchmark: >2.0x is typically considered safe
    """
    fixed_charges = income.interest_expense + lease_payments
    if fixed_charges == 0:
        return float('inf') if income.operating_income > 0 else 0.0

    ebit_plus_leases = income.operating_income + lease_payments
    return ebit_plus_leases / fixed_charges


def debt_service_coverage(
    cash_flow: CashFlowStatement,
    income: IncomeStatement,
    current_portion_lt_debt: float = 0.0
) -> float:
    """
    Debt Service Coverage = Operating Cash Flow / Total Debt Service

    Total Debt Service = Interest + Principal Payments

    Measures ability to service all debt obligations.

    Args:
        cash_flow: Cash flow statement
        income: Income statement (for interest expense)
        current_portion_lt_debt: Principal payments due in current year

    Benchmark:
    - >1.5x: Comfortable coverage
    - 1.0-1.5x: Tight but manageable
    - <1.0x: Cannot cover debt service from operations
    """
    debt_service = income.interest_expense + current_portion_lt_debt
    if debt_service == 0:
        return float('inf') if cash_flow.cash_from_operations > 0 else 0.0
    return cash_flow.cash_from_operations / debt_service


def cash_flow_to_debt(
    cash_flow: CashFlowStatement,
    balance: BalanceSheet
) -> float:
    """
    Cash Flow to Debt = Operating Cash Flow / Total Debt

    Measures ability to repay debt from cash flow.

    Benchmark:
    - >30%: Strong debt coverage
    - 20-30%: Adequate
    - <20%: May struggle to reduce debt
    """
    if balance.total_debt == 0:
        return float('inf') if cash_flow.cash_from_operations > 0 else 0.0
    return cash_flow.cash_from_operations / balance.total_debt


def debt_payback_years(
    income: IncomeStatement,
    balance: BalanceSheet
) -> float:
    """
    Debt Payback Period = Total Debt / EBITDA

    Years required to repay debt from EBITDA.

    Benchmark:
    - <3 years: Low risk
    - 3-5 years: Moderate
    - >5 years: High leverage
    """
    if income.ebitda <= 0:
        return float('inf') if balance.total_debt > 0 else 0.0
    return balance.total_debt / income.ebitda


# =============================================================================
# LEVERAGE RATIOS
# =============================================================================

def financial_leverage(balance: BalanceSheet) -> float:
    """
    Financial Leverage = Total Assets / Total Equity
    Also known as: Equity Multiplier

    Measures how many dollars of assets are controlled per dollar of equity.

    Benchmark:
    - 1.5-2.0: Conservative
    - 2.0-3.0: Moderate leverage
    - >3.0: High leverage

    Note: A value of 2.0 means 50% of assets are financed by equity
    """
    if balance.total_equity == 0:
        return float('inf') if balance.total_assets > 0 else 0.0
    return balance.total_assets / balance.total_equity


def equity_multiplier(balance: BalanceSheet) -> float:
    """
    Equity Multiplier = Total Assets / Total Equity

    Same as financial leverage; used in DuPont decomposition.

    ROE = ROA × Equity Multiplier
    """
    return financial_leverage(balance)


def capitalization_ratio(balance: BalanceSheet) -> float:
    """
    Capitalization Ratio = Long-term Debt / (LT Debt + Equity)

    Shows long-term debt as proportion of permanent capital.

    Benchmark: Similar to debt-to-capital but focuses on long-term
    """
    permanent_capital = balance.long_term_debt + balance.total_equity
    if permanent_capital == 0:
        return 0.0
    return balance.long_term_debt / permanent_capital


def degree_of_financial_leverage(
    income: IncomeStatement
) -> float:
    """
    Degree of Financial Leverage = EBIT / (EBIT - Interest)
    Also: % Change in EPS / % Change in EBIT

    Measures sensitivity of EPS to changes in EBIT.

    Interpretation:
    - DFL of 2.0 means 10% change in EBIT leads to 20% change in EPS
    - Higher DFL = more volatile earnings

    Benchmark: <2.0 is generally considered moderate
    """
    ebt = income.operating_income - income.interest_expense
    if ebt == 0:
        return float('inf') if income.operating_income != 0 else 0.0
    return income.operating_income / ebt


def equity_ratio(balance: BalanceSheet) -> float:
    """
    Equity Ratio = Total Equity / Total Assets

    Shows proportion of assets financed by shareholders.
    Inverse of financial leverage interpretation.

    Benchmark:
    - >50%: Conservative financing
    - 30-50%: Moderate
    - <30%: Aggressive use of leverage
    """
    if balance.total_assets == 0:
        return 0.0
    return balance.total_equity / balance.total_assets


def debt_to_tangible_net_worth(balance: BalanceSheet) -> float:
    """
    Debt to Tangible Net Worth = Total Debt / Tangible Net Worth

    Tangible Net Worth = Equity - Intangibles - Goodwill

    More conservative measure for lenders; excludes intangible assets.

    Benchmark: Lower is better; varies by industry
    """
    tangible_net_worth = balance.total_equity - balance.goodwill - balance.intangible_assets
    if tangible_net_worth <= 0:
        return float('inf') if balance.total_debt > 0 else 0.0
    return balance.total_debt / tangible_net_worth


# =============================================================================
# COMPREHENSIVE ANALYSIS CLASS
# =============================================================================

class SolvencyAnalysis:
    """
    Comprehensive solvency analysis.

    Calculates all solvency ratios for a company.

    Example:
        analysis = SolvencyAnalysis(statements)
        result = analysis.analyze()
        print(f"D/E: {result.debt_to_equity:.2f}")
        print(f"Interest Coverage: {result.interest_coverage:.1f}x")
    """

    def __init__(
        self,
        current: FinancialStatements,
        prior: Optional[FinancialStatements] = None,
        lease_payments: float = 0.0
    ):
        self.current = current
        self.prior = prior
        self.lease_payments = lease_payments

    def analyze(self) -> SolvencyResult:
        """Calculate all solvency ratios."""
        income = self.current.income_statement
        balance = self.current.balance_sheet
        cash_flow_stmt = self.current.cash_flow

        # Get current portion of LT debt for debt service calculation
        current_lt_debt = balance.current_portion_lt_debt

        return SolvencyResult(
            # Debt Ratios
            debt_to_equity=debt_to_equity(balance),
            debt_to_assets=debt_to_assets(balance),
            debt_to_capital=debt_to_capital(balance),
            long_term_debt_to_equity=long_term_debt_to_equity(balance),
            long_term_debt_to_capital=long_term_debt_to_capital(balance),
            net_debt_to_equity=net_debt_to_equity(balance),
            net_debt_to_ebitda=net_debt_to_ebitda(income, balance),

            # Coverage Ratios
            interest_coverage=interest_coverage(income),
            cash_interest_coverage=cash_interest_coverage(cash_flow_stmt, income),
            fixed_charge_coverage=fixed_charge_coverage(income, self.lease_payments),
            debt_service_coverage=debt_service_coverage(
                cash_flow_stmt, income, current_lt_debt
            ),
            cash_flow_to_debt=cash_flow_to_debt(cash_flow_stmt, balance),

            # Leverage Ratios
            financial_leverage=financial_leverage(balance),
            equity_multiplier=equity_multiplier(balance),
            capitalization_ratio=capitalization_ratio(balance),
            degree_of_financial_leverage=degree_of_financial_leverage(income),
        )

    def to_dict(self) -> Dict[str, float]:
        """Return all ratios as dictionary."""
        result = self.analyze()
        return {
            'debt_to_equity': result.debt_to_equity,
            'debt_to_assets': result.debt_to_assets,
            'debt_to_capital': result.debt_to_capital,
            'long_term_debt_to_equity': result.long_term_debt_to_equity,
            'long_term_debt_to_capital': result.long_term_debt_to_capital,
            'net_debt_to_equity': result.net_debt_to_equity,
            'net_debt_to_ebitda': result.net_debt_to_ebitda,
            'interest_coverage': result.interest_coverage,
            'cash_interest_coverage': result.cash_interest_coverage,
            'fixed_charge_coverage': result.fixed_charge_coverage,
            'debt_service_coverage': result.debt_service_coverage,
            'cash_flow_to_debt': result.cash_flow_to_debt,
            'financial_leverage': result.financial_leverage,
            'equity_multiplier': result.equity_multiplier,
            'capitalization_ratio': result.capitalization_ratio,
        }

    def assess_credit_risk(self) -> str:
        """Provide a qualitative assessment of credit risk."""
        result = self.analyze()

        issues = []
        strengths = []

        # Debt Level Assessment
        if result.debt_to_equity < 0.5:
            strengths.append("Very conservative debt levels")
        elif result.debt_to_equity < 1.0:
            strengths.append("Moderate debt levels")
        elif result.debt_to_equity < 2.0:
            issues.append("Above-average debt levels")
        else:
            issues.append("High debt levels; potential credit risk")

        # Net Debt/EBITDA Assessment
        if result.net_debt_to_ebitda < 1.5:
            strengths.append("Very low net debt relative to earnings")
        elif result.net_debt_to_ebitda < 3.0:
            strengths.append("Manageable net debt levels")
        elif result.net_debt_to_ebitda < 4.5:
            issues.append("Elevated net debt; monitor closely")
        else:
            issues.append("High net debt/EBITDA; refinancing risk")

        # Interest Coverage Assessment
        if result.interest_coverage >= 5.0:
            strengths.append("Strong interest coverage")
        elif result.interest_coverage >= 3.0:
            strengths.append("Adequate interest coverage")
        elif result.interest_coverage >= 2.0:
            issues.append("Interest coverage on the low side")
        elif result.interest_coverage >= 1.0:
            issues.append("Tight interest coverage; limited cushion")
        else:
            issues.append("Cannot cover interest expense; distress signal")

        # Cash Flow Coverage
        if result.cash_flow_to_debt > 0.3:
            strengths.append("Strong cash flow relative to debt")
        elif result.cash_flow_to_debt > 0.2:
            strengths.append("Adequate cash flow coverage")
        else:
            issues.append("Limited cash flow for debt reduction")

        assessment = "CREDIT RISK ASSESSMENT\n" + "=" * 40 + "\n\n"

        if strengths:
            assessment += "STRENGTHS:\n"
            for s in strengths:
                assessment += f"  + {s}\n"
            assessment += "\n"

        if issues:
            assessment += "CONCERNS:\n"
            for i in issues:
                assessment += f"  - {i}\n"

        # Overall rating
        score = len(strengths) - len(issues)
        if score >= 2:
            rating = "LOW RISK"
        elif score >= 0:
            rating = "MODERATE RISK"
        elif score >= -2:
            rating = "ELEVATED RISK"
        else:
            rating = "HIGH RISK"

        assessment += f"\nOVERALL CREDIT RISK: {rating}\n"

        return assessment
