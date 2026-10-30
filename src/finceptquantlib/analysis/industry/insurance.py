"""
Insurance Company Analysis

Specialized financial metrics for insurance companies:
- Underwriting Profitability (Loss Ratio, Expense Ratio, Combined Ratio)
- Investment Performance
- Reserve Adequacy
- Solvency and Capital Metrics
- Reinsurance Analysis
"""

from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class InsuranceFinancials:
    """Financial data structure for insurance companies."""

    # Premiums
    gross_written_premiums: float = 0.0
    net_written_premiums: float = 0.0
    earned_premiums: float = 0.0
    unearned_premium_reserve: float = 0.0

    # Underwriting
    losses_incurred: float = 0.0
    loss_adjustment_expenses: float = 0.0
    underwriting_expenses: float = 0.0
    policy_acquisition_costs: float = 0.0
    general_expenses: float = 0.0

    # Investment
    investment_income: float = 0.0
    realized_gains_losses: float = 0.0
    unrealized_gains_losses: float = 0.0
    invested_assets: float = 0.0

    # Reserves
    loss_reserves: float = 0.0
    prior_year_reserve_development: float = 0.0

    # Balance Sheet
    total_assets: float = 0.0
    total_liabilities: float = 0.0
    policyholders_surplus: float = 0.0  # Equity equivalent

    # Reinsurance
    ceded_premiums: float = 0.0
    reinsurance_recoverables: float = 0.0
    ceded_losses: float = 0.0

    # Other
    net_income: float = 0.0
    dividends_to_policyholders: float = 0.0
    shares_outstanding: float = 0.0

    @property
    def total_losses_and_lae(self) -> float:
        """Total losses and loss adjustment expenses."""
        return self.losses_incurred + self.loss_adjustment_expenses

    @property
    def total_underwriting_expense(self) -> float:
        """Total underwriting expenses."""
        return (self.underwriting_expenses + self.policy_acquisition_costs +
                self.general_expenses)

    @property
    def underwriting_result(self) -> float:
        """Underwriting profit/loss."""
        return (self.earned_premiums - self.total_losses_and_lae -
                self.total_underwriting_expense)


# =============================================================================
# Underwriting Ratios
# =============================================================================

def loss_ratio(
    losses_incurred: float,
    earned_premiums: float,
    include_lae: bool = True,
    loss_adjustment_expenses: float = 0.0
) -> Optional[float]:
    """
    Loss Ratio.

    Key underwriting profitability metric.

    Loss Ratio = (Losses + LAE) / Earned Premiums

    Benchmarks (P&C):
    - Excellent: < 55%
    - Good: 55% - 65%
    - Average: 65% - 75%
    - Poor: > 75%

    Args:
        losses_incurred: Net losses incurred
        earned_premiums: Net earned premiums
        include_lae: Whether to include loss adjustment expenses
        loss_adjustment_expenses: LAE amount

    Returns:
        Loss ratio as decimal
    """
    if earned_premiums <= 0:
        return None

    total_losses = losses_incurred
    if include_lae:
        total_losses += loss_adjustment_expenses

    return total_losses / earned_premiums


def expense_ratio(
    underwriting_expenses: float,
    written_premiums: float = None,
    earned_premiums: float = None,
    use_written: bool = True
) -> Optional[float]:
    """
    Expense Ratio.

    Operating efficiency of insurance operations.

    Trade basis: Expenses / Written Premiums
    GAAP basis: Expenses / Earned Premiums

    Benchmarks (P&C):
    - Excellent: < 25%
    - Good: 25% - 30%
    - Average: 30% - 35%
    - Poor: > 35%

    Args:
        underwriting_expenses: Total underwriting expenses
        written_premiums: Net written premiums (trade basis)
        earned_premiums: Net earned premiums (GAAP basis)
        use_written: Use written (True) or earned (False) premiums

    Returns:
        Expense ratio as decimal
    """
    denominator = written_premiums if use_written else earned_premiums
    if denominator is None or denominator <= 0:
        return None
    return underwriting_expenses / denominator


def combined_ratio(
    loss_ratio_value: float,
    expense_ratio_value: float,
    dividend_ratio: float = 0.0
) -> float:
    """
    Combined Ratio.

    Primary measure of underwriting profitability.

    Combined Ratio = Loss Ratio + Expense Ratio + Dividend Ratio

    Interpretation:
    - < 100%: Underwriting profit
    - = 100%: Break-even
    - > 100%: Underwriting loss

    Benchmarks:
    - Excellent: < 90%
    - Good: 90% - 95%
    - Average: 95% - 100%
    - Poor: > 100%

    Args:
        loss_ratio_value: Loss ratio
        expense_ratio_value: Expense ratio
        dividend_ratio: Policyholder dividend ratio (if applicable)

    Returns:
        Combined ratio as decimal
    """
    return loss_ratio_value + expense_ratio_value + dividend_ratio


def underwriting_margin(combined_ratio_value: float) -> float:
    """
    Underwriting Margin.

    Underwriting Margin = 1 - Combined Ratio

    Positive = profit, Negative = loss

    Args:
        combined_ratio_value: Combined ratio

    Returns:
        Underwriting margin as decimal
    """
    return 1.0 - combined_ratio_value


def operating_ratio(
    combined_ratio_value: float,
    investment_income_ratio: float
) -> float:
    """
    Operating Ratio.

    Overall insurance profitability including investment income.

    Operating Ratio = Combined Ratio - Investment Income Ratio

    Args:
        combined_ratio_value: Combined ratio
        investment_income_ratio: Investment income / Earned premiums

    Returns:
        Operating ratio as decimal
    """
    return combined_ratio_value - investment_income_ratio


# =============================================================================
# Investment Metrics
# =============================================================================

def investment_yield(
    net_investment_income: float,
    average_invested_assets: float
) -> Optional[float]:
    """
    Investment Yield.

    Return on invested assets (excluding gains/losses).

    Args:
        net_investment_income: Investment income net of expenses
        average_invested_assets: Average invested assets

    Returns:
        Yield as decimal
    """
    if average_invested_assets <= 0:
        return None
    return net_investment_income / average_invested_assets


def total_investment_return(
    net_investment_income: float,
    realized_gains: float,
    unrealized_gains: float,
    average_invested_assets: float
) -> Optional[float]:
    """
    Total Investment Return.

    Complete return including gains/losses.

    Args:
        net_investment_income: Investment income
        realized_gains: Realized capital gains/losses
        unrealized_gains: Unrealized capital gains/losses
        average_invested_assets: Average invested assets

    Returns:
        Total return as decimal
    """
    if average_invested_assets <= 0:
        return None
    total_return = net_investment_income + realized_gains + unrealized_gains
    return total_return / average_invested_assets


def investment_income_ratio(
    net_investment_income: float,
    earned_premiums: float
) -> Optional[float]:
    """
    Investment Income Ratio.

    Investment contribution relative to premiums.

    Args:
        net_investment_income: Net investment income
        earned_premiums: Net earned premiums

    Returns:
        Ratio as decimal
    """
    if earned_premiums <= 0:
        return None
    return net_investment_income / earned_premiums


# =============================================================================
# Reserve Metrics
# =============================================================================

def reserve_ratio(
    loss_reserves: float,
    earned_premiums: float
) -> Optional[float]:
    """
    Reserve Ratio.

    Reserve adequacy relative to premium volume.

    Args:
        loss_reserves: Total loss reserves
        earned_premiums: Net earned premiums

    Returns:
        Ratio as decimal
    """
    if earned_premiums <= 0:
        return None
    return loss_reserves / earned_premiums


def reserve_development(
    prior_year_development: float,
    prior_year_reserves: float
) -> Optional[float]:
    """
    Reserve Development Ratio.

    Measures accuracy of prior reserve estimates.

    Positive = unfavorable (reserves were inadequate)
    Negative = favorable (reserves were excessive)

    Args:
        prior_year_development: Change in prior year reserves
        prior_year_reserves: Beginning reserves from prior year

    Returns:
        Development ratio as decimal
    """
    if prior_year_reserves <= 0:
        return None
    return prior_year_development / prior_year_reserves


def reserve_to_surplus(
    loss_reserves: float,
    policyholders_surplus: float
) -> Optional[float]:
    """
    Reserve to Surplus Ratio.

    Leverage measure - reserves relative to capital.

    Benchmarks:
    - Conservative: < 150%
    - Moderate: 150% - 250%
    - Aggressive: > 250%

    Args:
        loss_reserves: Total loss reserves
        policyholders_surplus: Policyholders' surplus (equity)

    Returns:
        Ratio as decimal
    """
    if policyholders_surplus <= 0:
        return None
    return loss_reserves / policyholders_surplus


# =============================================================================
# Reinsurance Metrics
# =============================================================================

def ceded_reinsurance_ratio(
    ceded_premiums: float,
    gross_written_premiums: float
) -> Optional[float]:
    """
    Ceded Reinsurance Ratio.

    Proportion of risk transferred to reinsurers.

    Args:
        ceded_premiums: Premiums ceded to reinsurers
        gross_written_premiums: Gross written premiums

    Returns:
        Ratio as decimal
    """
    if gross_written_premiums <= 0:
        return None
    return ceded_premiums / gross_written_premiums


def retention_ratio(
    net_written_premiums: float,
    gross_written_premiums: float
) -> Optional[float]:
    """
    Retention Ratio.

    Proportion of risk retained (not ceded).

    Retention = 1 - Cession Ratio

    Args:
        net_written_premiums: Net written premiums
        gross_written_premiums: Gross written premiums

    Returns:
        Ratio as decimal
    """
    if gross_written_premiums <= 0:
        return None
    return net_written_premiums / gross_written_premiums


def reinsurance_recoverables_ratio(
    reinsurance_recoverables: float,
    policyholders_surplus: float
) -> Optional[float]:
    """
    Reinsurance Recoverables to Surplus.

    Credit risk from reinsurance dependence.

    Higher ratio = more dependent on reinsurer credit quality.

    Args:
        reinsurance_recoverables: Amounts due from reinsurers
        policyholders_surplus: Policyholders' surplus

    Returns:
        Ratio as decimal
    """
    if policyholders_surplus <= 0:
        return None
    return reinsurance_recoverables / policyholders_surplus


# =============================================================================
# Solvency Metrics
# =============================================================================

def premium_to_surplus(
    net_written_premiums: float,
    policyholders_surplus: float
) -> Optional[float]:
    """
    Premium to Surplus Ratio (Kenney Ratio).

    Classic leverage measure for P&C insurers.

    Benchmarks:
    - Conservative: < 1.0x
    - Moderate: 1.0x - 2.0x
    - Aggressive: > 2.0x
    - NAIC guideline: < 3.0x

    Args:
        net_written_premiums: Net written premiums
        policyholders_surplus: Policyholders' surplus

    Returns:
        Ratio as decimal (e.g., 1.5 = 150%)
    """
    if policyholders_surplus <= 0:
        return None
    return net_written_premiums / policyholders_surplus


def solvency_ratio_insurance(
    policyholders_surplus: float,
    total_liabilities: float
) -> Optional[float]:
    """
    Solvency Ratio.

    Capital adequacy measure.

    Args:
        policyholders_surplus: Policyholders' surplus
        total_liabilities: Total liabilities

    Returns:
        Ratio as decimal
    """
    if total_liabilities <= 0:
        return None
    return policyholders_surplus / total_liabilities


def net_leverage_ratio(
    net_written_premiums: float,
    loss_reserves: float,
    policyholders_surplus: float
) -> Optional[float]:
    """
    Net Leverage Ratio.

    Combined leverage from premiums and reserves.

    Net Leverage = (NWP + Loss Reserves) / Surplus

    Benchmarks:
    - Conservative: < 3.0x
    - Moderate: 3.0x - 5.0x
    - Aggressive: > 5.0x

    Args:
        net_written_premiums: Net written premiums
        loss_reserves: Total loss reserves
        policyholders_surplus: Policyholders' surplus

    Returns:
        Ratio as decimal
    """
    if policyholders_surplus <= 0:
        return None
    return (net_written_premiums + loss_reserves) / policyholders_surplus


# =============================================================================
# ROE Decomposition
# =============================================================================

def roe_decomposition_insurance(
    underwriting_margin: float,
    investment_income_ratio: float,
    leverage: float,
    tax_rate: float = 0.21
) -> Dict[str, float]:
    """
    Insurance ROE Decomposition.

    Decomposes ROE into underwriting, investment, and leverage components.

    ROE = [Underwriting Margin + (Investment Yield × Investment Leverage)] × (1 - Tax)

    Args:
        underwriting_margin: 1 - Combined Ratio
        investment_income_ratio: Investment income / Earned premiums
        leverage: Assets / Surplus
        tax_rate: Effective tax rate

    Returns:
        Dictionary with ROE components
    """
    # Simplified decomposition
    pre_tax_return = underwriting_margin + investment_income_ratio
    roe_from_underwriting = underwriting_margin * leverage * (1 - tax_rate)
    roe_from_investments = investment_income_ratio * leverage * (1 - tax_rate)
    total_roe = (roe_from_underwriting + roe_from_investments)

    return {
        'total_roe': total_roe,
        'roe_from_underwriting': roe_from_underwriting,
        'roe_from_investments': roe_from_investments,
        'pre_tax_operating_return': pre_tax_return,
        'leverage_multiplier': leverage,
        'tax_effect': 1 - tax_rate,
    }


# =============================================================================
# Insurance Analysis Class
# =============================================================================

@dataclass
class UnderwritingMetrics:
    """Underwriting performance summary."""
    loss_ratio: Optional[float]
    expense_ratio: Optional[float]
    combined_ratio: Optional[float]
    underwriting_margin: Optional[float]
    operating_ratio: Optional[float]
    assessment: str = ""


@dataclass
class ReserveMetrics:
    """Reserve analysis summary."""
    reserve_ratio: Optional[float]
    reserve_to_surplus: Optional[float]
    reserve_development: Optional[float]
    assessment: str = ""


@dataclass
class SolvencyMetrics:
    """Solvency analysis summary."""
    premium_to_surplus: Optional[float]
    net_leverage: Optional[float]
    solvency_ratio: Optional[float]
    assessment: str = ""


class InsuranceAnalysis:
    """
    Comprehensive insurance company analysis.

    Provides specialized analysis for P&C and life insurance companies.
    """

    def __init__(self, financials: InsuranceFinancials):
        """Initialize with insurance financials."""
        self.financials = financials

    def analyze_underwriting(self) -> UnderwritingMetrics:
        """
        Analyze underwriting performance.

        Returns:
            Underwriting metrics with assessment
        """
        f = self.financials

        lr = loss_ratio(
            f.losses_incurred,
            f.earned_premiums,
            include_lae=True,
            loss_adjustment_expenses=f.loss_adjustment_expenses
        )

        er = expense_ratio(
            f.total_underwriting_expense,
            written_premiums=f.net_written_premiums,
            use_written=True
        )

        cr = None
        um = None
        opr = None

        if lr is not None and er is not None:
            cr = combined_ratio(lr, er)
            um = underwriting_margin(cr)

            inv_ratio = investment_income_ratio(
                f.investment_income,
                f.earned_premiums
            )
            if inv_ratio is not None:
                opr = operating_ratio(cr, inv_ratio)

        # Assessment
        assessment_parts = []
        if cr is not None:
            if cr < 0.90:
                assessment_parts.append("Excellent underwriting (CR < 90%)")
            elif cr < 0.95:
                assessment_parts.append("Good underwriting (CR 90-95%)")
            elif cr < 1.0:
                assessment_parts.append("Marginal underwriting (CR 95-100%)")
            else:
                assessment_parts.append("Underwriting loss (CR > 100%)")

        if lr is not None:
            if lr > 0.75:
                assessment_parts.append("High loss ratio")
        if er is not None:
            if er > 0.35:
                assessment_parts.append("High expense ratio")

        return UnderwritingMetrics(
            loss_ratio=lr,
            expense_ratio=er,
            combined_ratio=cr,
            underwriting_margin=um,
            operating_ratio=opr,
            assessment=" | ".join(assessment_parts) if assessment_parts else "Unable to assess"
        )

    def analyze_reserves(self) -> ReserveMetrics:
        """
        Analyze reserve adequacy.

        Returns:
            Reserve metrics with assessment
        """
        f = self.financials

        rr = reserve_ratio(f.loss_reserves, f.earned_premiums)
        rts = reserve_to_surplus(f.loss_reserves, f.policyholders_surplus)
        rd = reserve_development(
            f.prior_year_reserve_development,
            f.loss_reserves - f.prior_year_reserve_development
        ) if f.loss_reserves > f.prior_year_reserve_development else None

        # Assessment
        assessment_parts = []

        if rts is not None:
            if rts > 2.5:
                assessment_parts.append("High reserve leverage")
            elif rts > 1.5:
                assessment_parts.append("Moderate reserve leverage")
            else:
                assessment_parts.append("Conservative reserve leverage")

        if rd is not None:
            if rd > 0.05:
                assessment_parts.append("Significant adverse development")
            elif rd > 0:
                assessment_parts.append("Minor adverse development")
            elif rd < -0.05:
                assessment_parts.append("Significant favorable development")
            else:
                assessment_parts.append("Minor favorable development")

        return ReserveMetrics(
            reserve_ratio=rr,
            reserve_to_surplus=rts,
            reserve_development=rd,
            assessment=" | ".join(assessment_parts) if assessment_parts else "Unable to assess"
        )

    def analyze_solvency(self) -> SolvencyMetrics:
        """
        Analyze solvency and capital adequacy.

        Returns:
            Solvency metrics with assessment
        """
        f = self.financials

        pts = premium_to_surplus(f.net_written_premiums, f.policyholders_surplus)
        nlr = net_leverage_ratio(
            f.net_written_premiums,
            f.loss_reserves,
            f.policyholders_surplus
        )
        sr = solvency_ratio_insurance(f.policyholders_surplus, f.total_liabilities)

        # Assessment
        assessment_parts = []

        if pts is not None:
            if pts > 3.0:
                assessment_parts.append("EXCEEDS NAIC GUIDELINE (P/S > 3x)")
            elif pts > 2.0:
                assessment_parts.append("Aggressive premium leverage")
            elif pts > 1.0:
                assessment_parts.append("Moderate premium leverage")
            else:
                assessment_parts.append("Conservative premium leverage")

        if nlr is not None:
            if nlr > 5.0:
                assessment_parts.append("High overall leverage")
            elif nlr > 3.0:
                assessment_parts.append("Moderate overall leverage")
            else:
                assessment_parts.append("Low overall leverage")

        return SolvencyMetrics(
            premium_to_surplus=pts,
            net_leverage=nlr,
            solvency_ratio=sr,
            assessment=" | ".join(assessment_parts) if assessment_parts else "Unable to assess"
        )

    def analyze_investments(self) -> Dict[str, Optional[float]]:
        """
        Analyze investment performance.

        Returns:
            Dictionary of investment metrics
        """
        f = self.financials

        return {
            'investment_yield': investment_yield(
                f.investment_income,
                f.invested_assets
            ),
            'total_return': total_investment_return(
                f.investment_income,
                f.realized_gains_losses,
                f.unrealized_gains_losses,
                f.invested_assets
            ),
            'investment_income_ratio': investment_income_ratio(
                f.investment_income,
                f.earned_premiums
            ),
            'invested_assets_ratio': (
                f.invested_assets / f.total_assets
                if f.total_assets > 0 else None
            ),
        }

    def analyze_reinsurance(self) -> Dict[str, Optional[float]]:
        """
        Analyze reinsurance usage.

        Returns:
            Dictionary of reinsurance metrics
        """
        f = self.financials

        return {
            'cession_ratio': ceded_reinsurance_ratio(
                f.ceded_premiums,
                f.gross_written_premiums
            ),
            'retention_ratio': retention_ratio(
                f.net_written_premiums,
                f.gross_written_premiums
            ),
            'recoverables_to_surplus': reinsurance_recoverables_ratio(
                f.reinsurance_recoverables,
                f.policyholders_surplus
            ),
        }

    def roe_analysis(self, tax_rate: float = 0.21) -> Dict[str, float]:
        """
        Decompose ROE into components.

        Args:
            tax_rate: Effective tax rate

        Returns:
            ROE decomposition
        """
        f = self.financials

        underwriting = self.analyze_underwriting()
        um = underwriting.underwriting_margin or 0

        inv_ratio = investment_income_ratio(
            f.investment_income,
            f.earned_premiums
        ) or 0

        leverage = (
            f.total_assets / f.policyholders_surplus
            if f.policyholders_surplus > 0 else 1.0
        )

        return roe_decomposition_insurance(um, inv_ratio, leverage, tax_rate)

    def full_analysis(self) -> Dict:
        """
        Perform complete insurance analysis.

        Returns:
            Dictionary with all analysis components
        """
        return {
            'underwriting': self.analyze_underwriting(),
            'reserves': self.analyze_reserves(),
            'solvency': self.analyze_solvency(),
            'investments': self.analyze_investments(),
            'reinsurance': self.analyze_reinsurance(),
            'roe_decomposition': self.roe_analysis(),
        }
