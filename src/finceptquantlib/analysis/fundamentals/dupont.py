"""
DuPont Analysis

Comprehensive DuPont decomposition:
- 3-Factor DuPont (Traditional)
- 5-Factor DuPont (Extended)
- Component Analysis
- Trend Decomposition

DuPont analysis decomposes ROE into its fundamental drivers.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
from .statements import (
    IncomeStatement, BalanceSheet, FinancialStatements
)


@dataclass
class DuPont3Result:
    """Result of 3-Factor DuPont analysis."""
    # The three factors
    net_profit_margin: float = 0.0      # Net Income / Revenue
    asset_turnover: float = 0.0         # Revenue / Assets
    equity_multiplier: float = 0.0      # Assets / Equity

    # ROE (product of the three factors)
    roe: float = 0.0

    # Verification
    roe_calculated: float = 0.0         # Direct NI/Equity calculation


@dataclass
class DuPont5Result:
    """Result of 5-Factor DuPont analysis."""
    # The five factors
    tax_burden: float = 0.0             # Net Income / EBT (1 - tax rate effect)
    interest_burden: float = 0.0        # EBT / EBIT
    ebit_margin: float = 0.0            # EBIT / Revenue (Operating margin)
    asset_turnover: float = 0.0         # Revenue / Assets
    equity_multiplier: float = 0.0      # Assets / Equity

    # ROE (product of the five factors)
    roe: float = 0.0

    # Derived measures
    operating_roe: float = 0.0          # Before tax/interest effects
    financial_leverage_effect: float = 0.0


@dataclass
class ROICDecomposition:
    """ROIC decomposition into margin and turnover."""
    nopat_margin: float = 0.0           # NOPAT / Revenue
    invested_capital_turnover: float = 0.0  # Revenue / Invested Capital
    roic: float = 0.0


# =============================================================================
# 3-FACTOR DUPONT ANALYSIS
# =============================================================================

def dupont_3_factor(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> DuPont3Result:
    """
    3-Factor DuPont Decomposition

    ROE = Net Profit Margin × Asset Turnover × Equity Multiplier
        = (Net Income / Revenue) × (Revenue / Assets) × (Assets / Equity)

    This decomposition shows how profitability (margin), efficiency (turnover),
    and leverage (multiplier) contribute to return on equity.

    Args:
        income: Income statement
        balance: Current balance sheet
        prior_balance: Prior balance sheet (for averaging)

    Returns: DuPont3Result with all components
    """
    # Calculate averages if prior balance available
    if prior_balance:
        avg_assets = (balance.total_assets + prior_balance.total_assets) / 2
        avg_equity = (balance.total_equity + prior_balance.total_equity) / 2
    else:
        avg_assets = balance.total_assets
        avg_equity = balance.total_equity

    # Net Profit Margin = Net Income / Revenue
    if income.revenue == 0:
        npm = 0.0
    else:
        npm = income.net_income / income.revenue

    # Asset Turnover = Revenue / Average Assets
    if avg_assets == 0:
        at = 0.0
    else:
        at = income.revenue / avg_assets

    # Equity Multiplier = Average Assets / Average Equity
    if avg_equity == 0:
        em = float('inf') if avg_assets > 0 else 0.0
    else:
        em = avg_assets / avg_equity

    # ROE = NPM × AT × EM
    roe_decomposed = npm * at * em

    # Direct ROE calculation for verification
    if avg_equity == 0:
        roe_direct = 0.0
    else:
        roe_direct = income.net_income / avg_equity

    return DuPont3Result(
        net_profit_margin=npm,
        asset_turnover=at,
        equity_multiplier=em,
        roe=roe_decomposed,
        roe_calculated=roe_direct
    )


# =============================================================================
# 5-FACTOR DUPONT ANALYSIS
# =============================================================================

def dupont_5_factor(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None
) -> DuPont5Result:
    """
    5-Factor DuPont Decomposition

    ROE = Tax Burden × Interest Burden × EBIT Margin × Asset Turnover × Equity Multiplier
        = (NI/EBT) × (EBT/EBIT) × (EBIT/Revenue) × (Revenue/Assets) × (Assets/Equity)

    This extended decomposition separates the effects of:
    1. Tax Burden: Impact of taxes on earnings
    2. Interest Burden: Impact of interest expense on earnings
    3. EBIT Margin: Operating profitability
    4. Asset Turnover: Asset efficiency
    5. Equity Multiplier: Financial leverage

    Args:
        income: Income statement
        balance: Current balance sheet
        prior_balance: Prior balance sheet (for averaging)

    Returns: DuPont5Result with all components
    """
    # Calculate averages if prior balance available
    if prior_balance:
        avg_assets = (balance.total_assets + prior_balance.total_assets) / 2
        avg_equity = (balance.total_equity + prior_balance.total_equity) / 2
    else:
        avg_assets = balance.total_assets
        avg_equity = balance.total_equity

    # Tax Burden = Net Income / Pre-tax Income
    # Represents (1 - effective tax rate) effect
    if income.pretax_income == 0:
        tax_burden = 0.0
    else:
        tax_burden = income.net_income / income.pretax_income

    # Interest Burden = Pre-tax Income / EBIT
    # Represents the effect of interest expense
    if income.operating_income == 0:
        interest_burden = 0.0
    else:
        interest_burden = income.pretax_income / income.operating_income

    # EBIT Margin = EBIT / Revenue (Operating Margin)
    if income.revenue == 0:
        ebit_margin = 0.0
    else:
        ebit_margin = income.operating_income / income.revenue

    # Asset Turnover = Revenue / Average Assets
    if avg_assets == 0:
        asset_turnover = 0.0
    else:
        asset_turnover = income.revenue / avg_assets

    # Equity Multiplier = Average Assets / Average Equity
    if avg_equity == 0:
        equity_multiplier = float('inf') if avg_assets > 0 else 0.0
    else:
        equity_multiplier = avg_assets / avg_equity

    # ROE = All 5 factors multiplied
    roe = tax_burden * interest_burden * ebit_margin * asset_turnover * equity_multiplier

    # Operating ROE (before tax and interest effects)
    operating_roe = ebit_margin * asset_turnover * equity_multiplier

    # Financial leverage effect
    # How much leverage contributes to ROE vs ROA
    if avg_assets == 0:
        roa = 0.0
    else:
        roa = income.net_income / avg_assets
    leverage_effect = roe - roa if roa != 0 else 0.0

    return DuPont5Result(
        tax_burden=tax_burden,
        interest_burden=interest_burden,
        ebit_margin=ebit_margin,
        asset_turnover=asset_turnover,
        equity_multiplier=equity_multiplier,
        roe=roe,
        operating_roe=operating_roe,
        financial_leverage_effect=leverage_effect
    )


# =============================================================================
# ROIC DECOMPOSITION
# =============================================================================

def roic_decomposition(
    income: IncomeStatement,
    balance: BalanceSheet,
    prior_balance: Optional[BalanceSheet] = None,
    tax_rate: Optional[float] = None
) -> ROICDecomposition:
    """
    ROIC Decomposition

    ROIC = NOPAT Margin × Invested Capital Turnover
         = (NOPAT / Revenue) × (Revenue / Invested Capital)

    Similar to DuPont for ROE, but for ROIC.
    Useful for analyzing value creation.

    Args:
        income: Income statement
        balance: Current balance sheet
        prior_balance: Prior balance sheet (for averaging)
        tax_rate: Tax rate to use (defaults to effective)

    Returns: ROICDecomposition with components
    """
    if tax_rate is None:
        tax_rate = income.effective_tax_rate

    # Calculate NOPAT
    nopat = income.operating_income * (1 - tax_rate)

    # Average invested capital
    if prior_balance:
        avg_ic = (balance.invested_capital + prior_balance.invested_capital) / 2
    else:
        avg_ic = balance.invested_capital

    # NOPAT Margin = NOPAT / Revenue
    if income.revenue == 0:
        nopat_margin = 0.0
    else:
        nopat_margin = nopat / income.revenue

    # Invested Capital Turnover = Revenue / Invested Capital
    if avg_ic == 0:
        ic_turnover = float('inf') if income.revenue > 0 else 0.0
    else:
        ic_turnover = income.revenue / avg_ic

    # ROIC = NOPAT Margin × IC Turnover
    roic = nopat_margin * ic_turnover

    return ROICDecomposition(
        nopat_margin=nopat_margin,
        invested_capital_turnover=ic_turnover,
        roic=roic
    )


# =============================================================================
# DUPONT TREND ANALYSIS
# =============================================================================

def dupont_trend_analysis(
    statements: List[FinancialStatements]
) -> List[DuPont5Result]:
    """
    Analyze DuPont decomposition over multiple periods.

    Args:
        statements: List of financial statements (oldest first)

    Returns: List of DuPont5Result for each period
    """
    results = []
    for i, stmt in enumerate(statements):
        prior = statements[i-1] if i > 0 else None
        prior_balance = prior.balance_sheet if prior else None
        result = dupont_5_factor(stmt.income_statement, stmt.balance_sheet, prior_balance)
        results.append(result)
    return results


def identify_roe_drivers(
    current: DuPont5Result,
    prior: DuPont5Result
) -> Dict[str, float]:
    """
    Identify which factors are driving ROE changes.

    Args:
        current: Current period DuPont result
        prior: Prior period DuPont result

    Returns: Dictionary with contribution of each factor to ROE change
    """
    roe_change = current.roe - prior.roe

    # Calculate contribution of each factor
    # Using marginal contribution analysis
    contributions = {}

    # Tax burden contribution
    avg_other = (
        (current.interest_burden + prior.interest_burden) / 2 *
        (current.ebit_margin + prior.ebit_margin) / 2 *
        (current.asset_turnover + prior.asset_turnover) / 2 *
        (current.equity_multiplier + prior.equity_multiplier) / 2
    )
    contributions['tax_burden'] = (current.tax_burden - prior.tax_burden) * avg_other

    # Interest burden contribution
    avg_other = (
        (current.tax_burden + prior.tax_burden) / 2 *
        (current.ebit_margin + prior.ebit_margin) / 2 *
        (current.asset_turnover + prior.asset_turnover) / 2 *
        (current.equity_multiplier + prior.equity_multiplier) / 2
    )
    contributions['interest_burden'] = (current.interest_burden - prior.interest_burden) * avg_other

    # EBIT margin contribution
    avg_other = (
        (current.tax_burden + prior.tax_burden) / 2 *
        (current.interest_burden + prior.interest_burden) / 2 *
        (current.asset_turnover + prior.asset_turnover) / 2 *
        (current.equity_multiplier + prior.equity_multiplier) / 2
    )
    contributions['ebit_margin'] = (current.ebit_margin - prior.ebit_margin) * avg_other

    # Asset turnover contribution
    avg_other = (
        (current.tax_burden + prior.tax_burden) / 2 *
        (current.interest_burden + prior.interest_burden) / 2 *
        (current.ebit_margin + prior.ebit_margin) / 2 *
        (current.equity_multiplier + prior.equity_multiplier) / 2
    )
    contributions['asset_turnover'] = (current.asset_turnover - prior.asset_turnover) * avg_other

    # Equity multiplier contribution
    avg_other = (
        (current.tax_burden + prior.tax_burden) / 2 *
        (current.interest_burden + prior.interest_burden) / 2 *
        (current.ebit_margin + prior.ebit_margin) / 2 *
        (current.asset_turnover + prior.asset_turnover) / 2
    )
    contributions['equity_multiplier'] = (current.equity_multiplier - prior.equity_multiplier) * avg_other

    contributions['total_change'] = roe_change

    return contributions


# =============================================================================
# COMPREHENSIVE ANALYSIS CLASS
# =============================================================================

class DuPontAnalysis:
    """
    Comprehensive DuPont analysis.

    Calculates both 3-factor and 5-factor decompositions.

    Example:
        analysis = DuPontAnalysis(statements, prior_statements)
        result_3 = analysis.three_factor()
        result_5 = analysis.five_factor()
        print(f"ROE: {result_5.roe:.2%}")
        print(f"  Tax Burden: {result_5.tax_burden:.2%}")
        print(f"  Interest Burden: {result_5.interest_burden:.2%}")
        print(f"  EBIT Margin: {result_5.ebit_margin:.2%}")
        print(f"  Asset Turnover: {result_5.asset_turnover:.2f}x")
        print(f"  Equity Multiplier: {result_5.equity_multiplier:.2f}x")
    """

    def __init__(
        self,
        current: FinancialStatements,
        prior: Optional[FinancialStatements] = None
    ):
        self.current = current
        self.prior = prior

    def three_factor(self) -> DuPont3Result:
        """Calculate 3-factor DuPont decomposition."""
        prior_balance = self.prior.balance_sheet if self.prior else None
        return dupont_3_factor(
            self.current.income_statement,
            self.current.balance_sheet,
            prior_balance
        )

    def five_factor(self) -> DuPont5Result:
        """Calculate 5-factor DuPont decomposition."""
        prior_balance = self.prior.balance_sheet if self.prior else None
        return dupont_5_factor(
            self.current.income_statement,
            self.current.balance_sheet,
            prior_balance
        )

    def roic(self, tax_rate: Optional[float] = None) -> ROICDecomposition:
        """Calculate ROIC decomposition."""
        prior_balance = self.prior.balance_sheet if self.prior else None
        return roic_decomposition(
            self.current.income_statement,
            self.current.balance_sheet,
            prior_balance,
            tax_rate
        )

    def to_dict(self) -> Dict[str, float]:
        """Return all decomposition values as dictionary."""
        result_5 = self.five_factor()
        roic_result = self.roic()

        return {
            # 5-Factor
            'tax_burden': result_5.tax_burden,
            'interest_burden': result_5.interest_burden,
            'ebit_margin': result_5.ebit_margin,
            'asset_turnover': result_5.asset_turnover,
            'equity_multiplier': result_5.equity_multiplier,
            'roe': result_5.roe,
            # ROIC
            'nopat_margin': roic_result.nopat_margin,
            'invested_capital_turnover': roic_result.invested_capital_turnover,
            'roic': roic_result.roic,
        }

    def format_decomposition(self) -> str:
        """Format DuPont decomposition as readable string."""
        result_3 = self.three_factor()
        result_5 = self.five_factor()

        output = "DUPONT ANALYSIS\n" + "=" * 50 + "\n\n"

        output += "3-FACTOR DECOMPOSITION:\n"
        output += "-" * 30 + "\n"
        output += f"  Net Profit Margin:    {result_3.net_profit_margin:>8.2%}\n"
        output += f"  × Asset Turnover:     {result_3.asset_turnover:>8.2f}x\n"
        output += f"  × Equity Multiplier:  {result_3.equity_multiplier:>8.2f}x\n"
        output += f"  = ROE:                {result_3.roe:>8.2%}\n\n"

        output += "5-FACTOR DECOMPOSITION:\n"
        output += "-" * 30 + "\n"
        output += f"  Tax Burden:           {result_5.tax_burden:>8.2%}\n"
        output += f"  × Interest Burden:    {result_5.interest_burden:>8.2%}\n"
        output += f"  × EBIT Margin:        {result_5.ebit_margin:>8.2%}\n"
        output += f"  × Asset Turnover:     {result_5.asset_turnover:>8.2f}x\n"
        output += f"  × Equity Multiplier:  {result_5.equity_multiplier:>8.2f}x\n"
        output += f"  = ROE:                {result_5.roe:>8.2%}\n\n"

        output += "INTERPRETATION:\n"
        output += "-" * 30 + "\n"

        if result_5.ebit_margin > 0.15:
            output += "  + Strong operating profitability\n"
        elif result_5.ebit_margin < 0.05:
            output += "  - Weak operating profitability\n"

        if result_5.asset_turnover > 1.0:
            output += "  + Efficient asset utilization\n"
        elif result_5.asset_turnover < 0.5:
            output += "  - Asset-heavy business model\n"

        if result_5.equity_multiplier > 3.0:
            output += "  ! High financial leverage\n"
        elif result_5.equity_multiplier < 1.5:
            output += "  + Conservative capital structure\n"

        if result_5.tax_burden < 0.7:
            output += "  - High effective tax rate\n"

        if result_5.interest_burden < 0.8:
            output += "  - Significant interest expense burden\n"

        return output
