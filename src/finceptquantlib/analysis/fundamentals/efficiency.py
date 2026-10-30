"""
Efficiency Ratios (Activity Ratios)

Comprehensive efficiency analysis with 15+ ratios:
- Turnover Ratios (Asset, Inventory, Receivables, Payables)
- Days Ratios (DIO, DSO, DPO)
- Working Capital Efficiency
- Fixed Asset Utilization

These ratios measure how effectively a company uses its assets.
"""

from dataclasses import dataclass
from typing import Optional, Dict
from .statements import (
    IncomeStatement, BalanceSheet, CashFlowStatement, FinancialStatements
)


@dataclass
class EfficiencyResult:
    """Result of efficiency analysis."""
    # Turnover Ratios
    asset_turnover: float = 0.0
    fixed_asset_turnover: float = 0.0
    inventory_turnover: float = 0.0
    receivables_turnover: float = 0.0
    payables_turnover: float = 0.0
    working_capital_turnover: float = 0.0
    equity_turnover: float = 0.0
    capital_employed_turnover: float = 0.0

    # Days Ratios
    days_inventory: float = 0.0
    days_receivables: float = 0.0
    days_payables: float = 0.0
    cash_conversion_cycle: float = 0.0
    operating_cycle: float = 0.0

    # Per-Asset Metrics
    revenue_per_asset: float = 0.0
    capex_to_revenue: float = 0.0
    capex_to_depreciation: float = 0.0


# =============================================================================
# ASSET TURNOVER RATIOS
# =============================================================================

def asset_turnover(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Asset Turnover = Revenue / Average Total Assets

    Measures how efficiently a company generates revenue from assets.
    Key component of DuPont decomposition.

    Benchmark:
    - Asset-light businesses (software): >1.0
    - Manufacturing: 0.5-1.0
    - Utilities/REITs: <0.5

    Higher is better; indicates more efficient asset utilization.
    """
    if prior_balance:
        avg_assets = (balance.total_assets + prior_balance.total_assets) / 2
    else:
        avg_assets = balance.total_assets

    if avg_assets == 0:
        return 0.0
    return income.revenue / avg_assets


def fixed_asset_turnover(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Fixed Asset Turnover = Revenue / Average Net PP&E

    Measures how effectively fixed assets generate revenue.

    Benchmark: Higher is better; varies by capital intensity
    - Capital-light: >10x
    - Average: 2-5x
    - Capital-intensive: <2x
    """
    if prior_balance:
        avg_ppe = (balance.net_ppe + prior_balance.net_ppe) / 2
    else:
        avg_ppe = balance.net_ppe

    if avg_ppe == 0:
        return float('inf') if income.revenue > 0 else 0.0
    return income.revenue / avg_ppe


def inventory_turnover(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Inventory Turnover = Cost of Goods Sold / Average Inventory

    Measures how many times inventory is sold and replaced.

    Benchmark:
    - Retail: 8-12x
    - Manufacturing: 4-8x
    - Higher is generally better (faster inventory movement)
    - Very high may indicate stockouts
    """
    if prior_balance:
        avg_inventory = (balance.inventory + prior_balance.inventory) / 2
    else:
        avg_inventory = balance.inventory

    if avg_inventory == 0:
        return float('inf') if income.cost_of_goods_sold > 0 else 0.0
    return income.cost_of_goods_sold / avg_inventory


def receivables_turnover(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Receivables Turnover = Revenue / Average Accounts Receivable

    Measures how quickly receivables are collected.

    Benchmark:
    - Higher is better (faster collection)
    - 8-12x is typical (30-45 day collection period)
    - Very high may indicate overly strict credit policies
    """
    if prior_balance:
        avg_ar = (balance.accounts_receivable + prior_balance.accounts_receivable) / 2
    else:
        avg_ar = balance.accounts_receivable

    if avg_ar == 0:
        return float('inf') if income.revenue > 0 else 0.0
    return income.revenue / avg_ar


def payables_turnover(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Payables Turnover = Cost of Goods Sold / Average Accounts Payable

    Measures how quickly a company pays suppliers.

    Benchmark:
    - Lower can be better (using supplier financing)
    - Too low may strain supplier relationships
    - 6-12x is typical
    """
    if prior_balance:
        avg_ap = (balance.accounts_payable + prior_balance.accounts_payable) / 2
    else:
        avg_ap = balance.accounts_payable

    if avg_ap == 0:
        return float('inf') if income.cost_of_goods_sold > 0 else 0.0
    return income.cost_of_goods_sold / avg_ap


def working_capital_turnover(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Working Capital Turnover = Revenue / Average Working Capital

    Measures how efficiently working capital generates revenue.

    Benchmark:
    - Higher is better (more revenue per unit of working capital)
    - Negative working capital can result in negative ratio
    """
    if prior_balance:
        avg_wc = (balance.working_capital + prior_balance.working_capital) / 2
    else:
        avg_wc = balance.working_capital

    if avg_wc == 0:
        return float('inf') if income.revenue > 0 else 0.0
    return income.revenue / avg_wc


def equity_turnover(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Equity Turnover = Revenue / Average Shareholders' Equity

    Measures how efficiently equity capital generates revenue.

    Higher ratio indicates more efficient use of equity.
    """
    if prior_balance:
        avg_equity = (balance.total_equity + prior_balance.total_equity) / 2
    else:
        avg_equity = balance.total_equity

    if avg_equity == 0:
        return float('inf') if income.revenue > 0 else 0.0
    return income.revenue / avg_equity


def capital_employed_turnover(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Capital Employed Turnover = Revenue / Average Capital Employed

    Capital Employed = Total Assets - Current Liabilities

    Measures revenue generation from permanent capital.
    """
    if prior_balance:
        avg_ce = (balance.capital_employed + prior_balance.capital_employed) / 2
    else:
        avg_ce = balance.capital_employed

    if avg_ce == 0:
        return float('inf') if income.revenue > 0 else 0.0
    return income.revenue / avg_ce


def invested_capital_turnover(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Invested Capital Turnover = Revenue / Average Invested Capital

    Invested Capital = Total Debt + Total Equity - Cash

    Key component of ROIC decomposition:
    ROIC = NOPAT Margin × Invested Capital Turnover
    """
    if prior_balance:
        avg_ic = (balance.invested_capital + prior_balance.invested_capital) / 2
    else:
        avg_ic = balance.invested_capital

    if avg_ic == 0:
        return float('inf') if income.revenue > 0 else 0.0
    return income.revenue / avg_ic


# =============================================================================
# DAYS RATIOS
# =============================================================================

def days_inventory(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Days Inventory Outstanding = 365 / Inventory Turnover
    Also known as: DIO, Days Inventory, Average Days to Sell Inventory

    Average days inventory is held before sale.

    Benchmark:
    - Lower is generally better (faster movement)
    - Retail: 30-45 days
    - Manufacturing: 60-90 days
    """
    turnover = inventory_turnover(income, balance, prior_balance)
    if turnover == 0:
        return float('inf')
    if turnover == float('inf'):
        return 0.0
    return 365 / turnover


def days_receivables(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Days Sales Outstanding = 365 / Receivables Turnover
    Also known as: DSO, Average Collection Period

    Average days to collect receivables.

    Benchmark:
    - Lower is better (faster collection)
    - Should be close to payment terms offered
    - 30-45 days is typical
    """
    turnover = receivables_turnover(income, balance, prior_balance)
    if turnover == 0:
        return float('inf')
    if turnover == float('inf'):
        return 0.0
    return 365 / turnover


def days_payables(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Days Payables Outstanding = 365 / Payables Turnover
    Also known as: DPO, Average Payment Period

    Average days to pay suppliers.

    Benchmark:
    - Higher can be beneficial (free financing)
    - Should align with supplier terms
    - 30-60 days typical
    """
    turnover = payables_turnover(income, balance, prior_balance)
    if turnover == 0:
        return float('inf')
    if turnover == float('inf'):
        return 0.0
    return 365 / turnover


# =============================================================================
# OPERATING CYCLE METRICS
# =============================================================================

def operating_cycle(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Operating Cycle = DIO + DSO

    Time from acquiring inventory to collecting cash from sale.

    Benchmark: Shorter is better; varies by industry
    """
    dio = days_inventory(income, balance, prior_balance)
    dso = days_receivables(income, balance, prior_balance)
    return dio + dso


def cash_conversion_cycle(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> float:
    """
    Cash Conversion Cycle = DIO + DSO - DPO

    Net time between cash outflow and cash inflow.

    Benchmark:
    - Lower is better
    - Negative CCC: Receive payment before paying suppliers
      (excellent working capital management, e.g., Amazon)
    """
    dio = days_inventory(income, balance, prior_balance)
    dso = days_receivables(income, balance, prior_balance)
    dpo = days_payables(income, balance, prior_balance)
    return dio + dso - dpo


# =============================================================================
# CAPITAL EXPENDITURE RATIOS
# =============================================================================

def capex_to_revenue(
    cash_flow: CashFlowStatement,
    income: IncomeStatement
) -> float:
    """
    CapEx to Revenue = Capital Expenditures / Revenue

    Measures capital intensity of revenue generation.

    Benchmark:
    - <5%: Asset-light business model
    - 5-15%: Typical
    - >15%: Capital-intensive
    """
    if income.revenue == 0:
        return 0.0
    return cash_flow.capex / income.revenue


def capex_to_depreciation(
    cash_flow: CashFlowStatement,
    income: IncomeStatement
) -> float:
    """
    CapEx to Depreciation = Capital Expenditures / Depreciation

    Measures whether company is investing enough to maintain asset base.

    Benchmark:
    - <1.0: May be under-investing; asset base declining
    - 1.0-1.5: Maintaining asset base
    - >1.5: Growing/upgrading asset base
    """
    depreciation = income.depreciation + income.amortization
    if depreciation == 0:
        return float('inf') if cash_flow.capex > 0 else 0.0
    return cash_flow.capex / depreciation


def capex_to_operating_cash_flow(
    cash_flow: CashFlowStatement
) -> float:
    """
    CapEx to Operating Cash Flow = Capital Expenditures / CFO

    Measures capital intensity relative to cash generation.

    Benchmark:
    - <50%: Strong free cash flow generation
    - 50-80%: Moderate
    - >80%: Limited free cash flow
    """
    if cash_flow.cash_from_operations == 0:
        return float('inf') if cash_flow.capex > 0 else 0.0
    return cash_flow.capex / cash_flow.cash_from_operations


def maintenance_capex_estimate(
    cash_flow: CashFlowStatement,
    income: IncomeStatement
) -> float:
    """
    Maintenance CapEx Estimate = Depreciation (approximation)

    A rough estimate of maintenance capex.
    Growth CapEx = Total CapEx - Maintenance CapEx

    Note: This is a simplification; actual maintenance capex
    requires detailed analysis.
    """
    return income.depreciation + income.amortization


# =============================================================================
# OTHER EFFICIENCY METRICS
# =============================================================================

def operating_efficiency(income: IncomeStatement) -> float:
    """
    Operating Efficiency = Operating Expenses / Revenue

    Measures operating cost structure.

    Benchmark: Lower is better; varies by industry
    """
    if income.revenue == 0:
        return 0.0
    return income.operating_expenses / income.revenue


def sga_to_revenue(income: IncomeStatement) -> float:
    """
    SG&A to Revenue = Selling, General & Admin Expenses / Revenue

    Measures administrative cost efficiency.

    Benchmark:
    - <20%: Efficient
    - 20-35%: Typical
    - >35%: May indicate overhead issues
    """
    if income.revenue == 0:
        return 0.0
    return income.sga_expenses / income.revenue


def rd_intensity(income: IncomeStatement) -> float:
    """
    R&D Intensity = R&D Expenses / Revenue

    Measures investment in innovation.

    Benchmark:
    - Tech companies: 10-25%
    - Pharma: 15-30%
    - Most industries: <5%
    """
    if income.revenue == 0:
        return 0.0
    return income.rd_expenses / income.revenue


def depreciation_rate(
    income: IncomeStatement,
    balance: BalanceSheet
) -> float:
    """
    Depreciation Rate = Depreciation / Gross PP&E

    Indicates aggressiveness of depreciation policy.

    Benchmark: Depends on asset mix and depreciation method
    """
    if balance.property_plant_equipment == 0:
        return 0.0
    return income.depreciation / balance.property_plant_equipment


# =============================================================================
# COMPREHENSIVE ANALYSIS CLASS
# =============================================================================

class EfficiencyAnalysis:
    """
    Comprehensive efficiency analysis.

    Calculates all efficiency/activity ratios for a company.

    Example:
        analysis = EfficiencyAnalysis(statements, prior_statements)
        result = analysis.analyze()
        print(f"Asset Turnover: {result.asset_turnover:.2f}x")
        print(f"Cash Conversion Cycle: {result.cash_conversion_cycle:.0f} days")
    """

    def __init__(
        self,
        current: FinancialStatements,
        prior: Optional[FinancialStatements] = None
    ):
        self.current = current
        self.prior = prior

    def analyze(self) -> EfficiencyResult:
        """Calculate all efficiency ratios."""
        income = self.current.income_statement
        balance = self.current.balance_sheet
        cash_flow_stmt = self.current.cash_flow
        prior_balance = self.prior.balance_sheet if self.prior else None

        # Calculate days ratios
        dio = days_inventory(income, balance, prior_balance)
        dso = days_receivables(income, balance, prior_balance)
        dpo = days_payables(income, balance, prior_balance)

        return EfficiencyResult(
            # Turnover Ratios
            asset_turnover=asset_turnover(income, balance, prior_balance),
            fixed_asset_turnover=fixed_asset_turnover(income, balance, prior_balance),
            inventory_turnover=inventory_turnover(income, balance, prior_balance),
            receivables_turnover=receivables_turnover(income, balance, prior_balance),
            payables_turnover=payables_turnover(income, balance, prior_balance),
            working_capital_turnover=working_capital_turnover(income, balance, prior_balance),
            equity_turnover=equity_turnover(income, balance, prior_balance),
            capital_employed_turnover=capital_employed_turnover(income, balance, prior_balance),

            # Days Ratios
            days_inventory=dio,
            days_receivables=dso,
            days_payables=dpo,
            cash_conversion_cycle=dio + dso - dpo,
            operating_cycle=dio + dso,

            # CapEx Metrics
            revenue_per_asset=asset_turnover(income, balance, prior_balance),
            capex_to_revenue=capex_to_revenue(cash_flow_stmt, income),
            capex_to_depreciation=capex_to_depreciation(cash_flow_stmt, income),
        )

    def to_dict(self) -> Dict[str, float]:
        """Return all ratios as dictionary."""
        result = self.analyze()
        return {
            'asset_turnover': result.asset_turnover,
            'fixed_asset_turnover': result.fixed_asset_turnover,
            'inventory_turnover': result.inventory_turnover,
            'receivables_turnover': result.receivables_turnover,
            'payables_turnover': result.payables_turnover,
            'working_capital_turnover': result.working_capital_turnover,
            'equity_turnover': result.equity_turnover,
            'capital_employed_turnover': result.capital_employed_turnover,
            'days_inventory': result.days_inventory,
            'days_receivables': result.days_receivables,
            'days_payables': result.days_payables,
            'cash_conversion_cycle': result.cash_conversion_cycle,
            'operating_cycle': result.operating_cycle,
            'capex_to_revenue': result.capex_to_revenue,
            'capex_to_depreciation': result.capex_to_depreciation,
        }

    def assess_operational_efficiency(self) -> str:
        """Provide a qualitative assessment of operational efficiency."""
        result = self.analyze()

        issues = []
        strengths = []

        # Asset Turnover Assessment
        if result.asset_turnover >= 1.5:
            strengths.append("High asset turnover indicates efficient asset use")
        elif result.asset_turnover >= 0.75:
            pass  # Average
        else:
            issues.append("Low asset turnover; assets may be underutilized")

        # Inventory Management
        if result.days_inventory <= 45:
            strengths.append("Efficient inventory management")
        elif result.days_inventory <= 90:
            pass  # Average
        else:
            issues.append("High inventory days; working capital tied up in inventory")

        # Receivables Collection
        if result.days_receivables <= 30:
            strengths.append("Fast receivables collection")
        elif result.days_receivables <= 45:
            pass  # Average
        else:
            issues.append("Slow receivables collection; credit risk or collection issues")

        # Cash Conversion Cycle
        if result.cash_conversion_cycle < 0:
            strengths.append("Negative CCC; exceptional working capital management")
        elif result.cash_conversion_cycle <= 60:
            strengths.append("Efficient cash conversion cycle")
        elif result.cash_conversion_cycle <= 90:
            pass  # Average
        else:
            issues.append("Long cash conversion cycle; working capital efficiency could improve")

        # CapEx Assessment
        if result.capex_to_depreciation >= 1.0:
            strengths.append("CapEx exceeds depreciation; investing for growth")
        else:
            issues.append("CapEx below depreciation; may be underinvesting")

        assessment = "OPERATIONAL EFFICIENCY ASSESSMENT\n" + "=" * 40 + "\n\n"

        if strengths:
            assessment += "STRENGTHS:\n"
            for s in strengths:
                assessment += f"  + {s}\n"
            assessment += "\n"

        if issues:
            assessment += "AREAS FOR IMPROVEMENT:\n"
            for i in issues:
                assessment += f"  - {i}\n"

        if len(strengths) > len(issues):
            assessment += "\nOVERALL: Above-average operational efficiency\n"
        elif len(issues) > len(strengths):
            assessment += "\nOVERALL: Operational efficiency could be improved\n"
        else:
            assessment += "\nOVERALL: Average operational efficiency\n"

        return assessment
