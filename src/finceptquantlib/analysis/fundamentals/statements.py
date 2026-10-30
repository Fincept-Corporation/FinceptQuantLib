"""
Financial Statement Data Structures

Core data structures for financial statements:
- Income Statement
- Balance Sheet
- Cash Flow Statement
- Market Data

These provide the foundation for all financial analysis.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import date
from enum import Enum


class ReportingFrequency(Enum):
    """Financial reporting frequency."""
    ANNUAL = "annual"
    QUARTERLY = "quarterly"
    TTM = "ttm"  # Trailing Twelve Months


class AccountingStandard(Enum):
    """Accounting standard used."""
    US_GAAP = "us_gaap"
    IFRS = "ifrs"


@dataclass
class IncomeStatement:
    """
    Income Statement (Profit & Loss Statement).

    Contains all income statement line items needed for financial analysis.
    All monetary values should be in the same currency.

    Attributes:
        period_end: End date of the reporting period
        revenue: Total revenue/sales
        cost_of_goods_sold: Direct costs of goods/services sold
        gross_profit: Revenue - COGS
        operating_expenses: Total operating expenses
        sga_expenses: Selling, general & administrative
        rd_expenses: Research & development
        depreciation: Depreciation expense
        amortization: Amortization expense
        operating_income: EBIT (Earnings Before Interest & Taxes)
        interest_expense: Interest paid on debt
        interest_income: Interest earned
        other_income: Non-operating income/expenses
        pretax_income: EBT (Earnings Before Taxes)
        income_tax: Tax expense
        net_income: Bottom line profit
        shares_outstanding: Weighted avg shares (basic)
        shares_diluted: Weighted avg shares (diluted)
        eps_basic: Basic earnings per share
        eps_diluted: Diluted earnings per share
        dividends_per_share: Dividends declared per share
    """
    period_end: Optional[date] = None
    frequency: ReportingFrequency = ReportingFrequency.ANNUAL

    # Revenue
    revenue: float = 0.0
    cost_of_revenue: float = 0.0  # Alias for COGS for service companies
    cost_of_goods_sold: float = 0.0

    # Gross Profit
    gross_profit: float = 0.0

    # Operating Expenses
    operating_expenses: float = 0.0
    sga_expenses: float = 0.0
    rd_expenses: float = 0.0
    depreciation: float = 0.0
    amortization: float = 0.0
    other_operating_expenses: float = 0.0

    # Operating Income
    operating_income: float = 0.0  # EBIT

    # Non-Operating
    interest_expense: float = 0.0
    interest_income: float = 0.0
    other_income: float = 0.0
    other_expense: float = 0.0

    # Pre-tax and Tax
    pretax_income: float = 0.0  # EBT
    income_tax: float = 0.0

    # Net Income
    net_income: float = 0.0
    net_income_common: float = 0.0  # After preferred dividends

    # Per Share Data
    shares_outstanding: float = 0.0
    shares_diluted: float = 0.0
    eps_basic: float = 0.0
    eps_diluted: float = 0.0
    dividends_per_share: float = 0.0

    def __post_init__(self):
        """Calculate derived fields if not provided."""
        # Auto-calculate gross profit if not provided
        if self.gross_profit == 0 and self.revenue > 0:
            cogs = self.cost_of_goods_sold or self.cost_of_revenue
            self.gross_profit = self.revenue - cogs

        # Auto-calculate EPS if not provided
        if self.eps_basic == 0 and self.shares_outstanding > 0:
            self.eps_basic = self.net_income / self.shares_outstanding

        if self.eps_diluted == 0 and self.shares_diluted > 0:
            self.eps_diluted = self.net_income / self.shares_diluted

    @property
    def ebit(self) -> float:
        """EBIT (Earnings Before Interest and Taxes)."""
        return self.operating_income

    @property
    def ebitda(self) -> float:
        """EBITDA (Earnings Before Interest, Taxes, Depreciation & Amortization)."""
        return self.operating_income + self.depreciation + self.amortization

    @property
    def ebt(self) -> float:
        """EBT (Earnings Before Taxes)."""
        return self.pretax_income

    @property
    def cogs(self) -> float:
        """Cost of Goods Sold."""
        return self.cost_of_goods_sold or self.cost_of_revenue

    @property
    def effective_tax_rate(self) -> float:
        """Effective tax rate."""
        if self.pretax_income == 0:
            return 0.0
        return self.income_tax / self.pretax_income

    @property
    def interest_expense_net(self) -> float:
        """Net interest expense."""
        return self.interest_expense - self.interest_income


@dataclass
class BalanceSheet:
    """
    Balance Sheet (Statement of Financial Position).

    Contains all balance sheet line items for financial analysis.

    Attributes:
        period_end: Date of balance sheet
        # Assets
        cash_and_equivalents: Cash and cash equivalents
        short_term_investments: Marketable securities
        accounts_receivable: Trade receivables
        inventory: Inventory
        prepaid_expenses: Prepaid expenses
        other_current_assets: Other current assets
        total_current_assets: Sum of current assets
        property_plant_equipment: Gross PP&E
        accumulated_depreciation: Accumulated depreciation
        net_ppe: Net property, plant & equipment
        goodwill: Goodwill from acquisitions
        intangible_assets: Other intangibles
        long_term_investments: LT investments
        deferred_tax_assets: Deferred tax assets
        other_assets: Other non-current assets
        total_assets: Total assets
        # Liabilities
        accounts_payable: Trade payables
        short_term_debt: ST borrowings
        current_portion_lt_debt: Current portion of LT debt
        accrued_liabilities: Accrued expenses
        deferred_revenue: Unearned revenue
        other_current_liabilities: Other current liabilities
        total_current_liabilities: Sum of current liabilities
        long_term_debt: LT borrowings
        deferred_tax_liabilities: Deferred tax liabilities
        pension_liabilities: Pension obligations
        other_liabilities: Other LT liabilities
        total_liabilities: Total liabilities
        # Equity
        common_stock: Common stock at par
        additional_paid_in_capital: APIC
        retained_earnings: Accumulated profits
        treasury_stock: Treasury stock (negative)
        accumulated_other_comprehensive_income: AOCI
        minority_interest: Non-controlling interest
        total_equity: Total shareholders' equity
    """
    period_end: Optional[date] = None

    # Current Assets
    cash_and_equivalents: float = 0.0
    short_term_investments: float = 0.0
    accounts_receivable: float = 0.0
    inventory: float = 0.0
    prepaid_expenses: float = 0.0
    other_current_assets: float = 0.0
    total_current_assets: float = 0.0

    # Non-Current Assets
    property_plant_equipment: float = 0.0
    accumulated_depreciation: float = 0.0
    net_ppe: float = 0.0
    goodwill: float = 0.0
    intangible_assets: float = 0.0
    long_term_investments: float = 0.0
    deferred_tax_assets: float = 0.0
    other_assets: float = 0.0
    total_assets: float = 0.0

    # Current Liabilities
    accounts_payable: float = 0.0
    short_term_debt: float = 0.0
    current_portion_lt_debt: float = 0.0
    accrued_liabilities: float = 0.0
    deferred_revenue: float = 0.0
    income_tax_payable: float = 0.0
    other_current_liabilities: float = 0.0
    total_current_liabilities: float = 0.0

    # Non-Current Liabilities
    long_term_debt: float = 0.0
    deferred_tax_liabilities: float = 0.0
    pension_liabilities: float = 0.0
    operating_lease_liabilities: float = 0.0
    other_liabilities: float = 0.0
    total_liabilities: float = 0.0

    # Shareholders' Equity
    common_stock: float = 0.0
    preferred_stock: float = 0.0
    additional_paid_in_capital: float = 0.0
    retained_earnings: float = 0.0
    treasury_stock: float = 0.0
    accumulated_other_comprehensive_income: float = 0.0
    minority_interest: float = 0.0
    total_equity: float = 0.0

    def __post_init__(self):
        """Calculate derived fields if not provided."""
        # Auto-calculate net PPE
        if self.net_ppe == 0 and self.property_plant_equipment > 0:
            self.net_ppe = self.property_plant_equipment - abs(self.accumulated_depreciation)

    @property
    def total_debt(self) -> float:
        """Total debt (short-term + long-term)."""
        return (self.short_term_debt + self.current_portion_lt_debt +
                self.long_term_debt)

    @property
    def net_debt(self) -> float:
        """Net debt (total debt - cash)."""
        return self.total_debt - self.cash_and_equivalents - self.short_term_investments

    @property
    def working_capital(self) -> float:
        """Working capital (current assets - current liabilities)."""
        return self.total_current_assets - self.total_current_liabilities

    @property
    def net_working_capital(self) -> float:
        """Net working capital (excluding cash and debt)."""
        current_assets_ex_cash = (self.accounts_receivable + self.inventory +
                                   self.prepaid_expenses + self.other_current_assets)
        current_liab_ex_debt = (self.accounts_payable + self.accrued_liabilities +
                                 self.deferred_revenue + self.other_current_liabilities)
        return current_assets_ex_cash - current_liab_ex_debt

    @property
    def tangible_assets(self) -> float:
        """Tangible assets (total assets - goodwill - intangibles)."""
        return self.total_assets - self.goodwill - self.intangible_assets

    @property
    def tangible_equity(self) -> float:
        """Tangible equity (total equity - goodwill - intangibles)."""
        return self.total_equity - self.goodwill - self.intangible_assets

    @property
    def book_value(self) -> float:
        """Book value (same as total equity)."""
        return self.total_equity

    @property
    def invested_capital(self) -> float:
        """Invested capital (equity + debt - cash)."""
        return self.total_equity + self.total_debt - self.cash_and_equivalents

    @property
    def capital_employed(self) -> float:
        """Capital employed (total assets - current liabilities)."""
        return self.total_assets - self.total_current_liabilities

    @property
    def net_operating_assets(self) -> float:
        """Net Operating Assets (NOA) for accruals analysis."""
        operating_assets = (self.total_assets - self.cash_and_equivalents -
                           self.short_term_investments)
        operating_liabilities = self.total_liabilities - self.total_debt
        return operating_assets - operating_liabilities


@dataclass
class CashFlowStatement:
    """
    Cash Flow Statement.

    Contains all cash flow line items for financial analysis.

    Attributes:
        period_end: End date of reporting period
        # Operating Activities
        net_income: Starting point (from income statement)
        depreciation_amortization: D&A add-back
        stock_based_compensation: SBC add-back
        deferred_taxes: Deferred tax adjustment
        change_in_receivables: Change in AR
        change_in_inventory: Change in inventory
        change_in_payables: Change in AP
        other_operating: Other adjustments
        cash_from_operations: CFO total
        # Investing Activities
        capital_expenditures: CapEx (negative)
        acquisitions: Business acquisitions
        investment_purchases: Securities purchased
        investment_sales: Securities sold
        other_investing: Other investing
        cash_from_investing: CFI total
        # Financing Activities
        debt_issued: New borrowings
        debt_repaid: Debt repayments
        shares_issued: Stock issuance
        shares_repurchased: Buybacks (negative)
        dividends_paid: Dividends (negative)
        other_financing: Other financing
        cash_from_financing: CFF total
        # Summary
        net_change_in_cash: Total change
        beginning_cash: Starting cash
        ending_cash: Ending cash
    """
    period_end: Optional[date] = None
    frequency: ReportingFrequency = ReportingFrequency.ANNUAL

    # Operating Activities
    net_income: float = 0.0
    depreciation_amortization: float = 0.0
    stock_based_compensation: float = 0.0
    deferred_taxes: float = 0.0
    impairment_charges: float = 0.0
    gain_loss_on_investments: float = 0.0
    gain_loss_on_sale_of_assets: float = 0.0
    change_in_receivables: float = 0.0
    change_in_inventory: float = 0.0
    change_in_payables: float = 0.0
    change_in_other_working_capital: float = 0.0
    other_operating: float = 0.0
    cash_from_operations: float = 0.0

    # Investing Activities
    capital_expenditures: float = 0.0
    acquisitions: float = 0.0
    divestitures: float = 0.0
    investment_purchases: float = 0.0
    investment_sales: float = 0.0
    other_investing: float = 0.0
    cash_from_investing: float = 0.0

    # Financing Activities
    debt_issued: float = 0.0
    debt_repaid: float = 0.0
    shares_issued: float = 0.0
    shares_repurchased: float = 0.0
    dividends_paid: float = 0.0
    other_financing: float = 0.0
    cash_from_financing: float = 0.0

    # Summary
    effect_of_fx: float = 0.0
    net_change_in_cash: float = 0.0
    beginning_cash: float = 0.0
    ending_cash: float = 0.0

    @property
    def free_cash_flow(self) -> float:
        """Free Cash Flow = CFO - CapEx."""
        return self.cash_from_operations - abs(self.capital_expenditures)

    @property
    def capex(self) -> float:
        """Capital expenditures (as positive number)."""
        return abs(self.capital_expenditures)

    @property
    def net_debt_change(self) -> float:
        """Net change in debt."""
        return self.debt_issued - abs(self.debt_repaid)

    @property
    def net_share_activity(self) -> float:
        """Net share issuance/buyback."""
        return self.shares_issued - abs(self.shares_repurchased)


@dataclass
class MarketData:
    """
    Market data for valuation analysis.

    Attributes:
        date: Date of market data
        stock_price: Current stock price
        market_cap: Market capitalization
        shares_outstanding: Shares outstanding
        enterprise_value: EV
        beta: Beta coefficient
        avg_volume: Average trading volume
        fifty_two_week_high: 52-week high
        fifty_two_week_low: 52-week low
    """
    date: Optional[date] = None
    stock_price: float = 0.0
    market_cap: float = 0.0
    shares_outstanding: float = 0.0
    enterprise_value: float = 0.0
    beta: float = 1.0
    avg_volume: float = 0.0
    fifty_two_week_high: float = 0.0
    fifty_two_week_low: float = 0.0

    def __post_init__(self):
        """Calculate derived fields."""
        if self.market_cap == 0 and self.stock_price > 0 and self.shares_outstanding > 0:
            self.market_cap = self.stock_price * self.shares_outstanding


@dataclass
class FinancialStatements:
    """
    Complete set of financial statements for a period.

    Bundles income statement, balance sheet, and cash flow statement
    for a single reporting period.
    """
    income_statement: IncomeStatement = field(default_factory=IncomeStatement)
    balance_sheet: BalanceSheet = field(default_factory=BalanceSheet)
    cash_flow: CashFlowStatement = field(default_factory=CashFlowStatement)
    market_data: Optional[MarketData] = None

    @property
    def period_end(self) -> Optional[date]:
        """Get the period end date."""
        return (self.income_statement.period_end or
                self.balance_sheet.period_end or
                self.cash_flow.period_end)

    @property
    def is_complete(self) -> bool:
        """Check if all three statements have data."""
        return (self.income_statement.revenue > 0 and
                self.balance_sheet.total_assets > 0 and
                self.cash_flow.cash_from_operations != 0)


@dataclass
class Company:
    """
    Company with historical financial statements.

    Stores multiple periods of financial data for trend analysis.

    Attributes:
        name: Company name
        ticker: Stock ticker symbol
        industry: Industry classification
        sector: Sector classification
        currency: Reporting currency
        accounting_standard: GAAP or IFRS
        statements: List of financial statements (chronological)
    """
    name: str = ""
    ticker: str = ""
    industry: str = ""
    sector: str = ""
    currency: str = "USD"
    accounting_standard: AccountingStandard = AccountingStandard.US_GAAP
    statements: List[FinancialStatements] = field(default_factory=list)

    @property
    def latest(self) -> Optional[FinancialStatements]:
        """Get the most recent financial statements."""
        if not self.statements:
            return None
        return self.statements[-1]

    @property
    def prior_year(self) -> Optional[FinancialStatements]:
        """Get prior year financial statements."""
        if len(self.statements) < 2:
            return None
        return self.statements[-2]

    def get_period(self, period_end: date) -> Optional[FinancialStatements]:
        """Get statements for a specific period."""
        for stmt in self.statements:
            if stmt.period_end == period_end:
                return stmt
        return None

    def add_statements(self, statements: FinancialStatements):
        """Add financial statements (maintains chronological order)."""
        self.statements.append(statements)
        self.statements.sort(key=lambda x: x.period_end or date.min)


# Factory functions for creating statements from dictionaries

def income_statement_from_dict(data: Dict[str, Any]) -> IncomeStatement:
    """Create IncomeStatement from dictionary."""
    return IncomeStatement(**{
        k: v for k, v in data.items()
        if k in IncomeStatement.__dataclass_fields__
    })


def balance_sheet_from_dict(data: Dict[str, Any]) -> BalanceSheet:
    """Create BalanceSheet from dictionary."""
    return BalanceSheet(**{
        k: v for k, v in data.items()
        if k in BalanceSheet.__dataclass_fields__
    })


def cash_flow_from_dict(data: Dict[str, Any]) -> CashFlowStatement:
    """Create CashFlowStatement from dictionary."""
    return CashFlowStatement(**{
        k: v for k, v in data.items()
        if k in CashFlowStatement.__dataclass_fields__
    })
