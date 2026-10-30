"""
Banking & Financial Services Analysis

Specialized financial metrics for banks and financial institutions:
- Net Interest Margin and Spread Analysis
- Asset Quality Metrics (NPL, Coverage)
- Capital Adequacy (Basel III ratios)
- Liquidity Ratios (LCR, NSFR)
- Efficiency and Profitability Metrics
"""

from dataclasses import dataclass
from typing import Optional, Dict, List


@dataclass
class BankingFinancials:
    """Financial data structure for banks and financial institutions."""

    # Interest Income/Expense
    interest_income: float = 0.0
    interest_expense: float = 0.0
    net_interest_income: float = 0.0

    # Non-Interest Income
    fee_income: float = 0.0
    trading_income: float = 0.0
    other_non_interest_income: float = 0.0

    # Expenses
    personnel_expense: float = 0.0
    other_operating_expense: float = 0.0
    loan_loss_provision: float = 0.0

    # Balance Sheet - Assets
    total_assets: float = 0.0
    earning_assets: float = 0.0
    total_loans: float = 0.0
    gross_loans: float = 0.0
    non_performing_loans: float = 0.0
    loan_loss_reserves: float = 0.0
    securities: float = 0.0
    cash_and_equivalents: float = 0.0

    # Balance Sheet - Liabilities
    total_deposits: float = 0.0
    demand_deposits: float = 0.0
    time_deposits: float = 0.0
    borrowed_funds: float = 0.0
    total_liabilities: float = 0.0

    # Capital
    total_equity: float = 0.0
    tier1_capital: float = 0.0
    tier2_capital: float = 0.0
    total_regulatory_capital: float = 0.0
    risk_weighted_assets: float = 0.0

    # Liquidity (Basel III)
    high_quality_liquid_assets: float = 0.0
    net_cash_outflows_30day: float = 0.0
    available_stable_funding: float = 0.0
    required_stable_funding: float = 0.0

    # Other
    net_income: float = 0.0
    shares_outstanding: float = 0.0

    def __post_init__(self):
        """Calculate derived fields if not provided."""
        if self.net_interest_income == 0.0 and self.interest_income > 0:
            self.net_interest_income = self.interest_income - self.interest_expense

    @property
    def total_operating_income(self) -> float:
        """Total operating income (net interest + non-interest)."""
        return (self.net_interest_income + self.fee_income +
                self.trading_income + self.other_non_interest_income)

    @property
    def total_operating_expense(self) -> float:
        """Total operating expenses."""
        return self.personnel_expense + self.other_operating_expense

    @property
    def non_interest_income(self) -> float:
        """Total non-interest income."""
        return self.fee_income + self.trading_income + self.other_non_interest_income

    @property
    def interest_bearing_liabilities(self) -> float:
        """Interest-bearing liabilities."""
        return self.time_deposits + self.borrowed_funds


# =============================================================================
# Net Interest Margin Analysis
# =============================================================================

def net_interest_margin(
    net_interest_income: float,
    average_earning_assets: float
) -> Optional[float]:
    """
    Net Interest Margin (NIM).

    Key profitability metric measuring spread earned on interest-bearing assets.

    NIM = Net Interest Income / Average Earning Assets

    Typical ranges:
    - US Commercial Banks: 2.5% - 4.0%
    - Investment Banks: 1.0% - 2.0%
    - Credit Card Banks: 8.0% - 12.0%

    Args:
        net_interest_income: Interest income minus interest expense
        average_earning_assets: Average loans, securities, and other earning assets

    Returns:
        NIM as decimal (e.g., 0.035 = 3.5%)
    """
    if average_earning_assets <= 0:
        return None
    return net_interest_income / average_earning_assets


def net_interest_spread(
    yield_on_earning_assets: float,
    cost_of_interest_bearing_liabilities: float
) -> float:
    """
    Net Interest Spread.

    Difference between asset yield and liability cost.

    Spread = Yield on Earning Assets - Cost of Interest-Bearing Liabilities

    Args:
        yield_on_earning_assets: Average yield earned on assets
        cost_of_interest_bearing_liabilities: Average cost of funding

    Returns:
        Spread as decimal
    """
    return yield_on_earning_assets - cost_of_interest_bearing_liabilities


def yield_on_earning_assets(
    interest_income: float,
    average_earning_assets: float
) -> Optional[float]:
    """
    Yield on Earning Assets.

    Average interest rate earned on loans and investments.

    Args:
        interest_income: Total interest and dividend income
        average_earning_assets: Average earning assets

    Returns:
        Yield as decimal
    """
    if average_earning_assets <= 0:
        return None
    return interest_income / average_earning_assets


def cost_of_funds(
    interest_expense: float,
    average_interest_bearing_liabilities: float
) -> Optional[float]:
    """
    Cost of Funds.

    Average interest rate paid on deposits and borrowings.

    Args:
        interest_expense: Total interest expense
        average_interest_bearing_liabilities: Average deposits and borrowings

    Returns:
        Cost as decimal
    """
    if average_interest_bearing_liabilities <= 0:
        return None
    return interest_expense / average_interest_bearing_liabilities


# =============================================================================
# Efficiency Metrics
# =============================================================================

def efficiency_ratio(
    operating_expense: float,
    operating_income: float
) -> Optional[float]:
    """
    Efficiency Ratio.

    Measures operating efficiency - lower is better.

    Efficiency Ratio = Operating Expense / Operating Income

    Benchmarks:
    - Excellent: < 50%
    - Good: 50% - 60%
    - Average: 60% - 70%
    - Poor: > 70%

    Args:
        operating_expense: Non-interest expense (excluding provisions)
        operating_income: Net interest income + non-interest income

    Returns:
        Efficiency ratio as decimal
    """
    if operating_income <= 0:
        return None
    return operating_expense / operating_income


def cost_to_income_ratio(
    total_costs: float,
    total_income: float
) -> Optional[float]:
    """
    Cost-to-Income Ratio (European metric).

    Similar to efficiency ratio but may include different items.

    Args:
        total_costs: All operating costs
        total_income: All operating income

    Returns:
        Ratio as decimal
    """
    if total_income <= 0:
        return None
    return total_costs / total_income


def fee_income_ratio(
    fee_income: float,
    total_operating_income: float
) -> Optional[float]:
    """
    Fee Income Ratio.

    Proportion of income from fees vs. interest.
    Higher ratio indicates more stable, less rate-sensitive income.

    Args:
        fee_income: Non-interest fee and commission income
        total_operating_income: Total operating revenue

    Returns:
        Ratio as decimal
    """
    if total_operating_income <= 0:
        return None
    return fee_income / total_operating_income


def trading_income_ratio(
    trading_income: float,
    total_operating_income: float
) -> Optional[float]:
    """
    Trading Income Ratio.

    Proportion of income from trading activities.
    Higher ratio indicates more volatile earnings.

    Args:
        trading_income: Net trading gains/losses
        total_operating_income: Total operating revenue

    Returns:
        Ratio as decimal
    """
    if total_operating_income <= 0:
        return None
    return trading_income / total_operating_income


# =============================================================================
# Asset Quality Metrics
# =============================================================================

def loan_to_deposit_ratio(
    total_loans: float,
    total_deposits: float
) -> Optional[float]:
    """
    Loan-to-Deposit Ratio (LDR).

    Measures liquidity and lending aggressiveness.

    Benchmarks:
    - Conservative: < 70%
    - Moderate: 70% - 90%
    - Aggressive: > 90%

    Args:
        total_loans: Net loans outstanding
        total_deposits: Total customer deposits

    Returns:
        LDR as decimal
    """
    if total_deposits <= 0:
        return None
    return total_loans / total_deposits


def non_performing_loan_ratio(
    non_performing_loans: float,
    gross_loans: float
) -> Optional[float]:
    """
    Non-Performing Loan (NPL) Ratio.

    Key asset quality metric.

    NPL Ratio = NPLs / Gross Loans

    Benchmarks:
    - Excellent: < 1%
    - Good: 1% - 2%
    - Concerning: 2% - 5%
    - Distressed: > 5%

    Args:
        non_performing_loans: Loans 90+ days past due or non-accrual
        gross_loans: Total loans before reserves

    Returns:
        NPL ratio as decimal
    """
    if gross_loans <= 0:
        return None
    return non_performing_loans / gross_loans


def loan_loss_provision_ratio(
    loan_loss_provision: float,
    average_loans: float
) -> Optional[float]:
    """
    Loan Loss Provision Ratio.

    Credit cost as percentage of loans.

    Args:
        loan_loss_provision: Provision for loan losses (P&L expense)
        average_loans: Average loans outstanding

    Returns:
        Provision ratio as decimal
    """
    if average_loans <= 0:
        return None
    return loan_loss_provision / average_loans


def coverage_ratio(
    loan_loss_reserves: float,
    non_performing_loans: float
) -> Optional[float]:
    """
    NPL Coverage Ratio.

    Measures adequacy of loan loss reserves.

    Coverage = Loan Loss Reserves / Non-Performing Loans

    Benchmarks:
    - Strong: > 150%
    - Adequate: 100% - 150%
    - Weak: < 100%

    Args:
        loan_loss_reserves: Allowance for loan losses (balance sheet)
        non_performing_loans: Total non-performing loans

    Returns:
        Coverage ratio as decimal (e.g., 1.5 = 150%)
    """
    if non_performing_loans <= 0:
        return None
    return loan_loss_reserves / non_performing_loans


def net_charge_off_ratio(
    gross_charge_offs: float,
    recoveries: float,
    average_loans: float
) -> Optional[float]:
    """
    Net Charge-Off Ratio.

    Actual credit losses as percentage of loans.

    Args:
        gross_charge_offs: Loans written off during period
        recoveries: Amounts recovered on previously charged-off loans
        average_loans: Average loans outstanding

    Returns:
        Net charge-off ratio as decimal
    """
    if average_loans <= 0:
        return None
    return (gross_charge_offs - recoveries) / average_loans


# =============================================================================
# Capital Adequacy (Basel III)
# =============================================================================

def tier1_capital_ratio(
    tier1_capital: float,
    risk_weighted_assets: float
) -> Optional[float]:
    """
    Tier 1 Capital Ratio (Basel III).

    Core capital adequacy measure.

    T1 Ratio = Tier 1 Capital / Risk-Weighted Assets

    Basel III Minimums:
    - Minimum: 6.0%
    - With Conservation Buffer: 8.5%
    - G-SIB Surcharge: +1.0% to +2.5%

    Args:
        tier1_capital: CET1 + Additional Tier 1
        risk_weighted_assets: Risk-weighted assets

    Returns:
        Ratio as decimal
    """
    if risk_weighted_assets <= 0:
        return None
    return tier1_capital / risk_weighted_assets


def total_capital_ratio(
    total_regulatory_capital: float,
    risk_weighted_assets: float
) -> Optional[float]:
    """
    Total Capital Ratio (Basel III).

    Total Capital Ratio = (Tier 1 + Tier 2) / RWA

    Basel III Minimum: 8.0% (10.5% with conservation buffer)

    Args:
        total_regulatory_capital: Tier 1 + Tier 2 capital
        risk_weighted_assets: Risk-weighted assets

    Returns:
        Ratio as decimal
    """
    if risk_weighted_assets <= 0:
        return None
    return total_regulatory_capital / risk_weighted_assets


def leverage_ratio_basel(
    tier1_capital: float,
    total_exposure: float
) -> Optional[float]:
    """
    Basel III Leverage Ratio.

    Non-risk-based capital measure.

    Leverage Ratio = Tier 1 Capital / Total Exposure

    Basel III Minimum: 3.0%

    Args:
        tier1_capital: Tier 1 capital
        total_exposure: On and off-balance sheet exposures

    Returns:
        Ratio as decimal
    """
    if total_exposure <= 0:
        return None
    return tier1_capital / total_exposure


def return_on_risk_weighted_assets(
    net_income: float,
    risk_weighted_assets: float
) -> Optional[float]:
    """
    Return on Risk-Weighted Assets (RoRWA).

    Risk-adjusted profitability measure.

    Args:
        net_income: Net income
        risk_weighted_assets: Risk-weighted assets

    Returns:
        RoRWA as decimal
    """
    if risk_weighted_assets <= 0:
        return None
    return net_income / risk_weighted_assets


# =============================================================================
# Liquidity Ratios (Basel III)
# =============================================================================

def liquidity_coverage_ratio(
    high_quality_liquid_assets: float,
    net_cash_outflows_30day: float
) -> Optional[float]:
    """
    Liquidity Coverage Ratio (LCR) - Basel III.

    Short-term liquidity measure.

    LCR = HQLA / Net Cash Outflows (30 days)

    Basel III Minimum: 100%

    Args:
        high_quality_liquid_assets: Cash + central bank reserves + govt bonds
        net_cash_outflows_30day: Expected net outflows in 30-day stress

    Returns:
        LCR as decimal (e.g., 1.2 = 120%)
    """
    if net_cash_outflows_30day <= 0:
        return None
    return high_quality_liquid_assets / net_cash_outflows_30day


def net_stable_funding_ratio(
    available_stable_funding: float,
    required_stable_funding: float
) -> Optional[float]:
    """
    Net Stable Funding Ratio (NSFR) - Basel III.

    Long-term liquidity measure.

    NSFR = Available Stable Funding / Required Stable Funding

    Basel III Minimum: 100%

    Args:
        available_stable_funding: Stable funding sources (weighted)
        required_stable_funding: Required funding for assets (weighted)

    Returns:
        NSFR as decimal
    """
    if required_stable_funding <= 0:
        return None
    return available_stable_funding / required_stable_funding


# =============================================================================
# Banking Analysis Class
# =============================================================================

@dataclass
class NIMDecomposition:
    """Net Interest Margin decomposition results."""
    nim: Optional[float]
    yield_on_assets: Optional[float]
    cost_of_funds: Optional[float]
    spread: Optional[float]
    volume_effect: Optional[float] = None
    rate_effect: Optional[float] = None


@dataclass
class AssetQualityMetrics:
    """Asset quality metrics summary."""
    npl_ratio: Optional[float]
    coverage_ratio: Optional[float]
    provision_ratio: Optional[float]
    loan_to_deposit: Optional[float]
    risk_assessment: str = ""


@dataclass
class CapitalMetrics:
    """Capital adequacy metrics summary."""
    tier1_ratio: Optional[float]
    total_capital_ratio: Optional[float]
    leverage_ratio: Optional[float]
    rorwa: Optional[float]
    regulatory_status: str = ""


class BankingAnalysis:
    """
    Comprehensive banking analysis.

    Provides specialized analysis for banks and financial institutions.
    """

    def __init__(self, financials: BankingFinancials):
        """Initialize with banking financials."""
        self.financials = financials

    def analyze_nim(
        self,
        prior_financials: Optional[BankingFinancials] = None
    ) -> NIMDecomposition:
        """
        Analyze Net Interest Margin and its components.

        Args:
            prior_financials: Previous period for change analysis

        Returns:
            NIM decomposition with yield, cost, and spread
        """
        f = self.financials

        nim = net_interest_margin(f.net_interest_income, f.earning_assets)
        asset_yield = yield_on_earning_assets(f.interest_income, f.earning_assets)
        funding_cost = cost_of_funds(
            f.interest_expense,
            f.interest_bearing_liabilities
        )

        spread = None
        if asset_yield is not None and funding_cost is not None:
            spread = asset_yield - funding_cost

        # Volume and rate effects if prior period provided
        volume_effect = None
        rate_effect = None

        if prior_financials:
            pf = prior_financials
            prior_nim = net_interest_margin(pf.net_interest_income, pf.earning_assets)

            if nim is not None and prior_nim is not None:
                # Simplified decomposition
                nim_change = nim - prior_nim

                # Rate effect approximation
                if f.earning_assets > 0 and pf.earning_assets > 0:
                    avg_assets = (f.earning_assets + pf.earning_assets) / 2
                    rate_effect = nim_change * avg_assets / f.earning_assets
                    volume_effect = nim_change - rate_effect

        return NIMDecomposition(
            nim=nim,
            yield_on_assets=asset_yield,
            cost_of_funds=funding_cost,
            spread=spread,
            volume_effect=volume_effect,
            rate_effect=rate_effect
        )

    def analyze_asset_quality(self) -> AssetQualityMetrics:
        """
        Analyze loan portfolio quality.

        Returns:
            Asset quality metrics with risk assessment
        """
        f = self.financials

        npl = non_performing_loan_ratio(f.non_performing_loans, f.gross_loans)
        coverage = coverage_ratio(f.loan_loss_reserves, f.non_performing_loans)
        provision = loan_loss_provision_ratio(f.loan_loss_provision, f.total_loans)
        ltd = loan_to_deposit_ratio(f.total_loans, f.total_deposits)

        # Risk assessment
        risk_factors = []

        if npl is not None:
            if npl > 0.05:
                risk_factors.append("High NPL ratio (>5%)")
            elif npl > 0.02:
                risk_factors.append("Elevated NPL ratio (>2%)")

        if coverage is not None:
            if coverage < 1.0:
                risk_factors.append("Inadequate NPL coverage (<100%)")
            elif coverage < 1.5:
                risk_factors.append("Moderate NPL coverage")

        if ltd is not None:
            if ltd > 1.0:
                risk_factors.append("Loan-to-deposit ratio exceeds 100%")
            elif ltd > 0.9:
                risk_factors.append("High loan-to-deposit ratio")

        if not risk_factors:
            assessment = "Strong asset quality"
        elif len(risk_factors) == 1:
            assessment = f"Moderate concern: {risk_factors[0]}"
        else:
            assessment = f"Multiple concerns: {'; '.join(risk_factors)}"

        return AssetQualityMetrics(
            npl_ratio=npl,
            coverage_ratio=coverage,
            provision_ratio=provision,
            loan_to_deposit=ltd,
            risk_assessment=assessment
        )

    def analyze_capital(self) -> CapitalMetrics:
        """
        Analyze capital adequacy.

        Returns:
            Capital metrics with regulatory status assessment
        """
        f = self.financials

        t1_ratio = tier1_capital_ratio(f.tier1_capital, f.risk_weighted_assets)
        total_cap = total_capital_ratio(
            f.total_regulatory_capital,
            f.risk_weighted_assets
        )
        leverage = leverage_ratio_basel(f.tier1_capital, f.total_assets)
        rorwa = return_on_risk_weighted_assets(f.net_income, f.risk_weighted_assets)

        # Regulatory assessment (Basel III minimums)
        status_parts = []

        if t1_ratio is not None:
            if t1_ratio >= 0.085:
                status_parts.append("Tier 1: Well-capitalized")
            elif t1_ratio >= 0.06:
                status_parts.append("Tier 1: Adequately capitalized")
            else:
                status_parts.append("Tier 1: UNDERCAPITALIZED")

        if total_cap is not None:
            if total_cap >= 0.105:
                status_parts.append("Total: Well-capitalized")
            elif total_cap >= 0.08:
                status_parts.append("Total: Adequately capitalized")
            else:
                status_parts.append("Total: UNDERCAPITALIZED")

        if leverage is not None:
            if leverage >= 0.05:
                status_parts.append("Leverage: Strong")
            elif leverage >= 0.03:
                status_parts.append("Leverage: Adequate")
            else:
                status_parts.append("Leverage: BELOW MINIMUM")

        return CapitalMetrics(
            tier1_ratio=t1_ratio,
            total_capital_ratio=total_cap,
            leverage_ratio=leverage,
            rorwa=rorwa,
            regulatory_status=" | ".join(status_parts) if status_parts else "Unable to assess"
        )

    def analyze_efficiency(self) -> Dict[str, Optional[float]]:
        """
        Analyze operational efficiency.

        Returns:
            Dictionary of efficiency metrics
        """
        f = self.financials

        return {
            'efficiency_ratio': efficiency_ratio(
                f.total_operating_expense,
                f.total_operating_income
            ),
            'cost_to_income': cost_to_income_ratio(
                f.total_operating_expense,
                f.total_operating_income
            ),
            'fee_income_ratio': fee_income_ratio(
                f.fee_income,
                f.total_operating_income
            ),
            'trading_income_ratio': trading_income_ratio(
                f.trading_income,
                f.total_operating_income
            ),
            'interest_income_ratio': (
                f.net_interest_income / f.total_operating_income
                if f.total_operating_income > 0 else None
            ),
        }

    def analyze_liquidity(self) -> Dict[str, Optional[float]]:
        """
        Analyze liquidity position.

        Returns:
            Dictionary of liquidity metrics
        """
        f = self.financials

        return {
            'lcr': liquidity_coverage_ratio(
                f.high_quality_liquid_assets,
                f.net_cash_outflows_30day
            ),
            'nsfr': net_stable_funding_ratio(
                f.available_stable_funding,
                f.required_stable_funding
            ),
            'loan_to_deposit': loan_to_deposit_ratio(
                f.total_loans,
                f.total_deposits
            ),
            'liquid_assets_ratio': (
                f.high_quality_liquid_assets / f.total_assets
                if f.total_assets > 0 else None
            ),
            'cash_to_deposits': (
                f.cash_and_equivalents / f.total_deposits
                if f.total_deposits > 0 else None
            ),
        }

    def full_analysis(self) -> Dict:
        """
        Perform complete banking analysis.

        Returns:
            Dictionary with all analysis components
        """
        return {
            'nim_analysis': self.analyze_nim(),
            'asset_quality': self.analyze_asset_quality(),
            'capital_adequacy': self.analyze_capital(),
            'efficiency': self.analyze_efficiency(),
            'liquidity': self.analyze_liquidity(),
        }

    def peer_comparison(
        self,
        peer_financials: List[BankingFinancials]
    ) -> Dict[str, Dict]:
        """
        Compare bank metrics against peers.

        Args:
            peer_financials: List of peer bank financials

        Returns:
            Comparison results with percentile rankings
        """
        # Calculate metrics for this bank
        self_metrics = {
            'nim': net_interest_margin(
                self.financials.net_interest_income,
                self.financials.earning_assets
            ),
            'efficiency': efficiency_ratio(
                self.financials.total_operating_expense,
                self.financials.total_operating_income
            ),
            'npl': non_performing_loan_ratio(
                self.financials.non_performing_loans,
                self.financials.gross_loans
            ),
            'tier1': tier1_capital_ratio(
                self.financials.tier1_capital,
                self.financials.risk_weighted_assets
            ),
        }

        # Calculate metrics for peers
        peer_metrics = {metric: [] for metric in self_metrics}

        for peer in peer_financials:
            peer_metrics['nim'].append(
                net_interest_margin(peer.net_interest_income, peer.earning_assets)
            )
            peer_metrics['efficiency'].append(
                efficiency_ratio(
                    peer.total_operating_expense,
                    peer.net_interest_income + peer.non_interest_income
                )
            )
            peer_metrics['npl'].append(
                non_performing_loan_ratio(peer.non_performing_loans, peer.gross_loans)
            )
            peer_metrics['tier1'].append(
                tier1_capital_ratio(peer.tier1_capital, peer.risk_weighted_assets)
            )

        # Calculate percentiles
        def percentile_rank(value: Optional[float], values: List) -> Optional[float]:
            if value is None:
                return None
            valid_values = [v for v in values if v is not None]
            if not valid_values:
                return None
            count_below = sum(1 for v in valid_values if v < value)
            return count_below / len(valid_values)

        comparisons = {}
        for metric, value in self_metrics.items():
            peers = [v for v in peer_metrics[metric] if v is not None]
            comparisons[metric] = {
                'value': value,
                'peer_median': sorted(peers)[len(peers)//2] if peers else None,
                'percentile': percentile_rank(value, peers),
            }

        return comparisons
