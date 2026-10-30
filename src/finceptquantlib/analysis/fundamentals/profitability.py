"""
Profitability Ratios

Comprehensive profitability analysis with 25+ ratios:
- Margin Ratios (Gross, Operating, EBITDA, Net, Pre-tax)
- Return Ratios (ROA, ROE, ROIC, ROCE, RONA)
- Per Employee Metrics
- Leverage Effect Decomposition

All ratios return decimals (e.g., 0.25 for 25%).
"""

from dataclasses import dataclass
from typing import Optional, Dict
from .statements import (
    IncomeStatement, BalanceSheet, CashFlowStatement, FinancialStatements
)


@dataclass
class ProfitabilityResult:
    """Result of profitability analysis."""
    # Margins
    gross_margin: float = 0.0
    operating_margin: float = 0.0
    ebitda_margin: float = 0.0
    pretax_margin: float = 0.0
    net_margin: float = 0.0

    # Returns
    return_on_assets: float = 0.0
    return_on_equity: float = 0.0
    return_on_invested_capital: float = 0.0
    return_on_capital_employed: float = 0.0
    return_on_tangible_assets: float = 0.0
    return_on_net_assets: float = 0.0
    cash_return_on_assets: float = 0.0
    cash_return_on_invested_capital: float = 0.0

    # Leverage Components
    tax_burden: float = 0.0
    interest_burden: float = 0.0
    operating_leverage: float = 0.0
    financial_leverage: float = 0.0

    # Other
    effective_tax_rate: float = 0.0
    basic_earning_power: float = 0.0
    retention_rate: float = 0.0
    payout_ratio: float = 0.0


# =============================================================================
# MARGIN RATIOS
# =============================================================================

def gross_margin(income: IncomeStatement) -> float:
    """
    Gross Profit Margin = Gross Profit / Revenue

    Measures the percentage of revenue retained after direct costs.
    Higher is better; varies significantly by industry.

    Benchmark: Varies by industry (20-70%+ typical)
    """
    if income.revenue == 0:
        return 0.0
    return income.gross_profit / income.revenue


def operating_margin(income: IncomeStatement) -> float:
    """
    Operating Profit Margin = Operating Income / Revenue
    Also known as: EBIT Margin

    Measures profitability from core operations before
    interest and taxes.

    Benchmark: 10-20% is healthy for most industries
    """
    if income.revenue == 0:
        return 0.0
    return income.operating_income / income.revenue


def ebitda_margin(income: IncomeStatement) -> float:
    """
    EBITDA Margin = EBITDA / Revenue

    Measures operating cash generation potential.
    Useful for comparing companies with different capital structures.

    Benchmark: 15-25% is typical
    """
    if income.revenue == 0:
        return 0.0
    return income.ebitda / income.revenue


def pretax_margin(income: IncomeStatement) -> float:
    """
    Pre-tax Profit Margin = EBT / Revenue

    Measures profitability before tax effects.

    Benchmark: 5-15% typical
    """
    if income.revenue == 0:
        return 0.0
    return income.pretax_income / income.revenue


def net_margin(income: IncomeStatement) -> float:
    """
    Net Profit Margin = Net Income / Revenue

    Measures the percentage of revenue that becomes profit.
    The ultimate profitability measure.

    Benchmark: 5-15% typical, varies by industry
    """
    if income.revenue == 0:
        return 0.0
    return income.net_income / income.revenue


def core_operating_margin(income: IncomeStatement) -> float:
    """
    Core Operating Margin = (Operating Income - Other Income) / Revenue

    Excludes non-recurring items for cleaner operating view.
    """
    if income.revenue == 0:
        return 0.0
    core_ebit = income.operating_income - income.other_income
    return core_ebit / income.revenue


# =============================================================================
# RETURN RATIOS
# =============================================================================

def return_on_assets(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Return on Assets = Net Income / Average Total Assets

    Measures how efficiently a company uses its assets to generate profit.

    Args:
        income: Current period income statement
        balance: Current period balance sheet
        prior_balance: Prior period balance sheet (for average)

    Benchmark: 5-10% typical, higher for asset-light businesses
    """
    if prior_balance:
        avg_assets = (balance.total_assets + prior_balance.total_assets) / 2
    else:
        avg_assets = balance.total_assets

    if avg_assets == 0:
        return 0.0
    return income.net_income / avg_assets


def return_on_equity(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Return on Equity = Net Income / Average Shareholders' Equity

    Measures return generated on shareholders' investment.
    The most important return metric for equity investors.

    Benchmark: 15-20% is excellent, >10% is good
    """
    if prior_balance:
        avg_equity = (balance.total_equity + prior_balance.total_equity) / 2
    else:
        avg_equity = balance.total_equity

    if avg_equity == 0:
        return 0.0
    return income.net_income / avg_equity


def return_on_invested_capital(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None,
    tax_rate: Optional[float] = None
) -> float:
    """
    Return on Invested Capital = NOPAT / Invested Capital

    NOPAT = Operating Income × (1 - Tax Rate)
    Invested Capital = Total Debt + Total Equity - Cash

    The key metric for value creation: ROIC > WACC creates value.

    Benchmark: Compare to WACC; >10% is typically good
    """
    # Calculate NOPAT
    if tax_rate is None:
        tax_rate = income.effective_tax_rate
    nopat = income.operating_income * (1 - tax_rate)

    # Calculate average invested capital
    if prior_balance:
        avg_ic = (balance.invested_capital + prior_balance.invested_capital) / 2
    else:
        avg_ic = balance.invested_capital

    if avg_ic == 0:
        return 0.0
    return nopat / avg_ic


def return_on_capital_employed(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Return on Capital Employed = EBIT / Capital Employed

    Capital Employed = Total Assets - Current Liabilities

    Measures efficiency of capital usage including debt.

    Benchmark: >15% is typically good
    """
    if prior_balance:
        avg_ce = (balance.capital_employed + prior_balance.capital_employed) / 2
    else:
        avg_ce = balance.capital_employed

    if avg_ce == 0:
        return 0.0
    return income.operating_income / avg_ce


def return_on_tangible_assets(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Return on Tangible Assets = Net Income / Average Tangible Assets

    Excludes goodwill and intangibles for a more conservative view.
    """
    if prior_balance:
        avg_tangible = (balance.tangible_assets + prior_balance.tangible_assets) / 2
    else:
        avg_tangible = balance.tangible_assets

    if avg_tangible == 0:
        return 0.0
    return income.net_income / avg_tangible


def return_on_net_assets(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Return on Net Assets = Net Income / (Fixed Assets + Working Capital)

    Also known as RONA.
    """
    net_assets = balance.net_ppe + balance.working_capital
    if prior_balance:
        prior_net_assets = prior_balance.net_ppe + prior_balance.working_capital
        avg_net_assets = (net_assets + prior_net_assets) / 2
    else:
        avg_net_assets = net_assets

    if avg_net_assets == 0:
        return 0.0
    return income.net_income / avg_net_assets


def cash_return_on_assets(
    cash_flow: CashFlowStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Cash Return on Assets = Operating Cash Flow / Average Total Assets

    Cash-based alternative to ROA.
    """
    if prior_balance:
        avg_assets = (balance.total_assets + prior_balance.total_assets) / 2
    else:
        avg_assets = balance.total_assets

    if avg_assets == 0:
        return 0.0
    return cash_flow.cash_from_operations / avg_assets


def cash_return_on_invested_capital(
    cash_flow: CashFlowStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Cash Return on Invested Capital = Operating Cash Flow / Invested Capital

    Cash-based alternative to ROIC.
    """
    if prior_balance:
        avg_ic = (balance.invested_capital + prior_balance.invested_capital) / 2
    else:
        avg_ic = balance.invested_capital

    if avg_ic == 0:
        return 0.0
    return cash_flow.cash_from_operations / avg_ic


# =============================================================================
# LEVERAGE EFFECT COMPONENTS
# =============================================================================

def tax_burden(income: IncomeStatement) -> float:
    """
    Tax Burden = Net Income / Pre-tax Income

    Component of 5-factor DuPont.
    Measures the proportion of pre-tax income retained after taxes.

    Value of 1.0 = no taxes; lower values mean higher tax burden.
    """
    if income.pretax_income == 0:
        return 0.0
    return income.net_income / income.pretax_income


def interest_burden(income: IncomeStatement) -> float:
    """
    Interest Burden = Pre-tax Income / Operating Income

    Component of 5-factor DuPont.
    Measures the impact of interest expense on earnings.

    Value of 1.0 = no interest expense; lower values mean higher interest burden.
    """
    if income.operating_income == 0:
        return 0.0
    return income.pretax_income / income.operating_income


def effective_tax_rate(income: IncomeStatement) -> float:
    """
    Effective Tax Rate = Income Tax / Pre-tax Income

    The actual tax rate paid by the company.

    Benchmark: Compare to statutory rate (21% US corporate)
    """
    if income.pretax_income == 0:
        return 0.0
    return income.income_tax / income.pretax_income


def basic_earning_power(income: IncomeStatement, balance: BalanceSheet) -> float:
    """
    Basic Earning Power = EBIT / Total Assets

    Measures the raw earning power of assets before interest and taxes.
    """
    if balance.total_assets == 0:
        return 0.0
    return income.operating_income / balance.total_assets


# =============================================================================
# OTHER PROFITABILITY METRICS
# =============================================================================

def retention_rate(income: IncomeStatement) -> float:
    """
    Retention Rate = (Net Income - Dividends) / Net Income
                   = 1 - Payout Ratio

    The proportion of earnings retained for reinvestment.
    """
    if income.net_income == 0:
        return 0.0

    dividends = income.dividends_per_share * income.shares_outstanding
    return (income.net_income - dividends) / income.net_income


def payout_ratio(income: IncomeStatement) -> float:
    """
    Payout Ratio = Dividends / Net Income

    The proportion of earnings paid out as dividends.

    Benchmark: 30-50% typical for dividend-paying companies
    """
    if income.net_income == 0:
        return 0.0

    dividends = income.dividends_per_share * income.shares_outstanding
    return dividends / income.net_income


def revenue_per_employee(income: IncomeStatement, employees: int) -> float:
    """
    Revenue per Employee = Revenue / Number of Employees

    Measures labor productivity.
    """
    if employees == 0:
        return 0.0
    return income.revenue / employees


def profit_per_employee(income: IncomeStatement, employees: int) -> float:
    """
    Profit per Employee = Net Income / Number of Employees

    Measures profit efficiency per worker.
    """
    if employees == 0:
        return 0.0
    return income.net_income / employees


# =============================================================================
# COMPREHENSIVE ANALYSIS CLASS
# =============================================================================

class ProfitabilityAnalysis:
    """
    Comprehensive profitability analysis.

    Calculates all profitability ratios for a company.

    Example:
        analysis = ProfitabilityAnalysis(statements, prior_statements)
        result = analysis.analyze()
        print(f"ROE: {result.return_on_equity:.2%}")
    """

    def __init__(
        self,
        current: FinancialStatements,
        prior: Optional[FinancialStatements] = None
    ):
        self.current = current
        self.prior = prior

    def analyze(self) -> ProfitabilityResult:
        """Calculate all profitability ratios."""
        income = self.current.income_statement
        balance = self.current.balance_sheet
        cash_flow = self.current.cash_flow
        prior_balance = self.prior.balance_sheet if self.prior else None

        return ProfitabilityResult(
            # Margins
            gross_margin=gross_margin(income),
            operating_margin=operating_margin(income),
            ebitda_margin=ebitda_margin(income),
            pretax_margin=pretax_margin(income),
            net_margin=net_margin(income),

            # Returns
            return_on_assets=return_on_assets(income, balance, prior_balance),
            return_on_equity=return_on_equity(income, balance, prior_balance),
            return_on_invested_capital=return_on_invested_capital(
                income, balance, prior_balance
            ),
            return_on_capital_employed=return_on_capital_employed(
                income, balance, prior_balance
            ),
            return_on_tangible_assets=return_on_tangible_assets(
                income, balance, prior_balance
            ),
            return_on_net_assets=return_on_net_assets(
                income, balance, prior_balance
            ),
            cash_return_on_assets=cash_return_on_assets(
                cash_flow, balance, prior_balance
            ),
            cash_return_on_invested_capital=cash_return_on_invested_capital(
                cash_flow, balance, prior_balance
            ),

            # Leverage Components
            tax_burden=tax_burden(income),
            interest_burden=interest_burden(income),
            effective_tax_rate=effective_tax_rate(income),
            basic_earning_power=basic_earning_power(income, balance),

            # Other
            retention_rate=retention_rate(income),
            payout_ratio=payout_ratio(income),
        )

    def to_dict(self) -> Dict[str, float]:
        """Return all ratios as dictionary."""
        result = self.analyze()
        return {
            'gross_margin': result.gross_margin,
            'operating_margin': result.operating_margin,
            'ebitda_margin': result.ebitda_margin,
            'pretax_margin': result.pretax_margin,
            'net_margin': result.net_margin,
            'return_on_assets': result.return_on_assets,
            'return_on_equity': result.return_on_equity,
            'return_on_invested_capital': result.return_on_invested_capital,
            'return_on_capital_employed': result.return_on_capital_employed,
            'return_on_tangible_assets': result.return_on_tangible_assets,
            'effective_tax_rate': result.effective_tax_rate,
            'retention_rate': result.retention_rate,
            'payout_ratio': result.payout_ratio,
        }
