"""
Pro-Forma Adjustments Module

Standardize financial statements for better comparability:
- Operating Lease Capitalization
- R&D Capitalization
- Pension Adjustment
- Non-Recurring Items Normalization
- Goodwill and Intangibles Adjustment
- Stock-Based Compensation Adjustment
- LIFO to FIFO Conversion
- Off-Balance Sheet Items
"""

from dataclasses import dataclass
from typing import Optional, Dict, List


# =============================================================================
# Operating Lease Capitalization (Pre-IFRS 16 / ASC 842)
# =============================================================================

@dataclass
class LeaseCapitalizationResult:
    """Result of operating lease capitalization."""
    lease_asset: float
    lease_liability: float
    depreciation_adjustment: float
    interest_adjustment: float
    rent_expense_removed: float
    ebit_adjustment: float
    debt_adjustment: float
    asset_adjustment: float


def capitalize_operating_leases(
    annual_lease_payments: List[float],
    discount_rate: float = 0.05,
    remaining_years: Optional[int] = None
) -> LeaseCapitalizationResult:
    """
    Capitalize Operating Leases.

    Converts operating leases to finance lease treatment:
    - Add lease asset and liability to balance sheet
    - Replace rent expense with depreciation + interest

    Under IFRS 16 / ASC 842, this is now required. This function
    is useful for adjusting older financials or analyzing lease impact.

    Args:
        annual_lease_payments: Future minimum lease payments by year
        discount_rate: Incremental borrowing rate
        remaining_years: If payments list is incomplete, assume this many years

    Returns:
        LeaseCapitalizationResult with adjustments
    """
    if not annual_lease_payments:
        return LeaseCapitalizationResult(0, 0, 0, 0, 0, 0, 0, 0)

    # If remaining_years provided and longer than payment list, extend
    if remaining_years and remaining_years > len(annual_lease_payments):
        last_payment = annual_lease_payments[-1]
        for _ in range(remaining_years - len(annual_lease_payments)):
            annual_lease_payments.append(last_payment)

    # Calculate present value of lease payments
    pv_lease = 0.0
    for t, payment in enumerate(annual_lease_payments, 1):
        pv_lease += payment / ((1 + discount_rate) ** t)

    # Lease asset = Lease liability (at inception)
    lease_asset = pv_lease
    lease_liability = pv_lease

    # Year 1 adjustments
    current_year_rent = annual_lease_payments[0] if annual_lease_payments else 0

    # Depreciation (straight-line over lease term)
    lease_term = len(annual_lease_payments)
    depreciation = lease_asset / lease_term if lease_term > 0 else 0

    # Interest expense (Year 1)
    interest = lease_liability * discount_rate

    # EBIT adjustment: Remove rent, add back depreciation (interest is below EBIT)
    # Old: Rent reduces EBIT
    # New: Depreciation reduces EBIT, Interest reduces pre-tax income
    # EBIT change = +Rent - Depreciation = Rent - Depreciation
    ebit_adjustment = current_year_rent - depreciation

    return LeaseCapitalizationResult(
        lease_asset=lease_asset,
        lease_liability=lease_liability,
        depreciation_adjustment=depreciation,
        interest_adjustment=interest,
        rent_expense_removed=current_year_rent,
        ebit_adjustment=ebit_adjustment,
        debt_adjustment=lease_liability,
        asset_adjustment=lease_asset,
    )


def quick_lease_capitalization(
    current_year_rent: float,
    discount_rate: float = 0.05,
    remaining_lease_term: float = 8.0
) -> LeaseCapitalizationResult:
    """
    Quick Operating Lease Capitalization.

    Uses current rent expense to estimate lease liability.
    Common approximation: Liability ≈ Rent × Multiple

    Multiple ≈ [1 - (1+r)^(-n)] / r (PV annuity factor)

    Args:
        current_year_rent: Current year rent/lease expense
        discount_rate: Incremental borrowing rate
        remaining_lease_term: Estimated remaining lease term

    Returns:
        LeaseCapitalizationResult with adjustments
    """
    # PV of annuity factor
    if discount_rate > 0:
        pv_factor = (1 - (1 + discount_rate) ** (-remaining_lease_term)) / discount_rate
    else:
        pv_factor = remaining_lease_term

    lease_liability = current_year_rent * pv_factor
    lease_asset = lease_liability

    # Depreciation
    depreciation = lease_asset / remaining_lease_term

    # Interest
    interest = lease_liability * discount_rate

    # EBIT adjustment
    ebit_adjustment = current_year_rent - depreciation

    return LeaseCapitalizationResult(
        lease_asset=lease_asset,
        lease_liability=lease_liability,
        depreciation_adjustment=depreciation,
        interest_adjustment=interest,
        rent_expense_removed=current_year_rent,
        ebit_adjustment=ebit_adjustment,
        debt_adjustment=lease_liability,
        asset_adjustment=lease_asset,
    )


# =============================================================================
# R&D Capitalization
# =============================================================================

@dataclass
class RDCapitalizationResult:
    """Result of R&D capitalization."""
    rd_asset: float
    current_year_amortization: float
    rd_expense_removed: float
    net_income_adjustment: float
    equity_adjustment: float
    roic_adjustment: float


def capitalize_rd(
    rd_expenses: List[float],
    amortization_period: int = 5,
    tax_rate: float = 0.25
) -> RDCapitalizationResult:
    """
    Capitalize R&D Expenses.

    Treats R&D as an investment rather than expense:
    - Create R&D asset on balance sheet
    - Amortize over useful life
    - Adjust income and equity

    Args:
        rd_expenses: R&D expenses for past years (oldest first)
        amortization_period: Years to amortize R&D
        tax_rate: Tax rate for after-tax adjustments

    Returns:
        RDCapitalizationResult with adjustments
    """
    if not rd_expenses:
        return RDCapitalizationResult(0, 0, 0, 0, 0, 0)

    # Calculate unamortized R&D asset
    # Each year's R&D is amortized over the amortization period
    rd_asset = 0.0
    current_amortization = 0.0

    for i, rd_exp in enumerate(rd_expenses):
        years_since = len(rd_expenses) - i - 1  # Years since this R&D was incurred
        years_remaining = amortization_period - years_since

        if years_remaining > 0:
            # This R&D still has unamortized value
            unamortized = rd_exp * (years_remaining / amortization_period)
            rd_asset += unamortized

            # Amortization from this year's R&D
            annual_amort = rd_exp / amortization_period
            current_amortization += annual_amort

    # Current year's R&D (most recent)
    current_rd_expense = rd_expenses[-1] if rd_expenses else 0

    # Net income adjustment
    # Old: R&D fully expensed
    # New: R&D capitalized, only amortization expensed
    pretax_adjustment = current_rd_expense - current_amortization
    net_income_adjustment = pretax_adjustment * (1 - tax_rate)

    # Equity adjustment (cumulative effect)
    equity_adjustment = rd_asset * (1 - tax_rate)

    # ROIC adjustment
    # Numerator increases by net income adjustment
    # Denominator increases by R&D asset
    roic_adjustment = net_income_adjustment - (rd_asset * 0.10)  # Assuming 10% WACC

    return RDCapitalizationResult(
        rd_asset=rd_asset,
        current_year_amortization=current_amortization,
        rd_expense_removed=current_rd_expense,
        net_income_adjustment=net_income_adjustment,
        equity_adjustment=equity_adjustment,
        roic_adjustment=roic_adjustment,
    )


def rd_intensity_adjustment(
    rd_expense: float,
    revenue: float,
    industry_average_rd_intensity: float
) -> Dict[str, float]:
    """
    Adjust for R&D Intensity Differences.

    Normalizes earnings for companies with different R&D spending.

    Args:
        rd_expense: Current R&D expense
        revenue: Current revenue
        industry_average_rd_intensity: Industry average R&D/Revenue

    Returns:
        Dictionary with adjustment factors
    """
    actual_intensity = rd_expense / revenue if revenue > 0 else 0
    excess_rd = (actual_intensity - industry_average_rd_intensity) * revenue

    return {
        'actual_rd_intensity': actual_intensity,
        'industry_rd_intensity': industry_average_rd_intensity,
        'excess_rd': excess_rd,
        'normalized_rd': industry_average_rd_intensity * revenue,
        'earnings_adjustment': excess_rd,  # Add back excess R&D
    }


# =============================================================================
# Pension Adjustment
# =============================================================================

@dataclass
class PensionAdjustmentResult:
    """Result of pension adjustment."""
    pension_asset_or_liability: float
    service_cost: float
    interest_cost: float
    expected_return_on_assets: float
    total_pension_expense: float
    debt_adjustment: float
    operating_income_adjustment: float


def adjust_pension(
    projected_benefit_obligation: float,
    fair_value_plan_assets: float,
    service_cost: float,
    interest_cost: float,
    expected_return_on_assets: float,
    reported_pension_expense: float,
    tax_rate: float = 0.25
) -> PensionAdjustmentResult:
    """
    Adjust for Pension Accounting.

    Reclassifies pension expense components:
    - Service cost → Operating expense
    - Interest cost → Interest expense
    - Expected return → Investment income

    Adjusts balance sheet for funded status.

    Args:
        projected_benefit_obligation: PBO
        fair_value_plan_assets: Fair value of plan assets
        service_cost: Current service cost
        interest_cost: Interest cost on PBO
        expected_return_on_assets: Expected return on plan assets
        reported_pension_expense: Total reported pension expense
        tax_rate: Tax rate

    Returns:
        PensionAdjustmentResult with adjustments
    """
    # Funded status (liability if negative)
    funded_status = fair_value_plan_assets - projected_benefit_obligation
    pension_liability = max(0, -funded_status)

    # True pension expense (economic)
    economic_pension_expense = service_cost + interest_cost - expected_return_on_assets

    # Operating income adjustment
    # Only service cost is operating; rest is financial
    # Old: All pension expense in operating income
    # New: Only service cost in operating income
    operating_adjustment = reported_pension_expense - service_cost

    return PensionAdjustmentResult(
        pension_asset_or_liability=funded_status,
        service_cost=service_cost,
        interest_cost=interest_cost,
        expected_return_on_assets=expected_return_on_assets,
        total_pension_expense=economic_pension_expense,
        debt_adjustment=pension_liability,
        operating_income_adjustment=operating_adjustment,
    )


# =============================================================================
# Non-Recurring Items
# =============================================================================

@dataclass
class NormalizedEarningsResult:
    """Result of earnings normalization."""
    reported_earnings: float
    normalized_earnings: float
    total_adjustments: float
    adjustment_details: Dict[str, float]
    quality_score: float


def normalize_earnings(
    reported_net_income: float,
    restructuring_charges: float = 0.0,
    impairments: float = 0.0,
    litigation_settlements: float = 0.0,
    gains_losses_on_sales: float = 0.0,
    discontinued_operations: float = 0.0,
    tax_adjustments: float = 0.0,
    other_nonrecurring: float = 0.0,
    tax_rate: float = 0.25
) -> NormalizedEarningsResult:
    """
    Normalize Earnings for Non-Recurring Items.

    Removes one-time items to estimate sustainable earnings power.

    Args:
        reported_net_income: Reported net income
        restructuring_charges: Restructuring charges (positive = expense)
        impairments: Asset impairments
        litigation_settlements: Litigation gains/losses
        gains_losses_on_sales: Gains/losses on asset sales
        discontinued_operations: Discontinued operations income/loss
        tax_adjustments: One-time tax items
        other_nonrecurring: Other non-recurring items

    Returns:
        NormalizedEarningsResult with adjusted earnings
    """
    # Adjustments (positive = add back to earnings)
    adjustments = {
        'restructuring': restructuring_charges * (1 - tax_rate),
        'impairments': impairments * (1 - tax_rate),
        'litigation': -litigation_settlements * (1 - tax_rate),  # Negative if gain
        'asset_sales': -gains_losses_on_sales * (1 - tax_rate),
        'discontinued': -discontinued_operations,
        'tax_adjustments': -tax_adjustments,
        'other': other_nonrecurring * (1 - tax_rate),
    }

    total_adjustments = sum(adjustments.values())
    normalized = reported_net_income + total_adjustments

    # Quality score (0-100)
    # Higher score = less adjustments needed = higher quality
    adjustment_ratio = abs(total_adjustments) / abs(reported_net_income) if reported_net_income != 0 else 1
    quality_score = max(0, 100 * (1 - adjustment_ratio))

    return NormalizedEarningsResult(
        reported_earnings=reported_net_income,
        normalized_earnings=normalized,
        total_adjustments=total_adjustments,
        adjustment_details=adjustments,
        quality_score=quality_score,
    )


def identify_nonrecurring_items(
    current_expense: float,
    historical_average: float,
    historical_std: float,
    threshold_std: float = 2.0
) -> Dict[str, float]:
    """
    Identify Potential Non-Recurring Items.

    Flags expenses that deviate significantly from historical patterns.

    Args:
        current_expense: Current period expense
        historical_average: Historical average expense
        historical_std: Historical standard deviation
        threshold_std: Number of std deviations for flagging

    Returns:
        Dictionary with analysis
    """
    if historical_std <= 0:
        z_score = 0
    else:
        z_score = (current_expense - historical_average) / historical_std

    is_anomaly = abs(z_score) > threshold_std
    anomaly_amount = current_expense - historical_average if is_anomaly else 0

    return {
        'current': current_expense,
        'historical_average': historical_average,
        'z_score': z_score,
        'is_anomaly': is_anomaly,
        'anomaly_amount': anomaly_amount,
        'normalized_estimate': historical_average if is_anomaly else current_expense,
    }


# =============================================================================
# Stock-Based Compensation
# =============================================================================

def adjust_stock_compensation(
    reported_income: float,
    stock_comp_expense: float,
    diluted_shares: float,
    basic_shares: float,
    stock_price: float,
    tax_rate: float = 0.25
) -> Dict[str, float]:
    """
    Adjust for Stock-Based Compensation.

    SBC is a real economic cost but non-cash.
    Different views on treatment:
    1. Keep as expense (standard GAAP)
    2. Add back as non-cash
    3. Adjust for dilution effect

    Args:
        reported_income: Reported net income (after SBC)
        stock_comp_expense: Stock-based compensation expense
        diluted_shares: Diluted share count
        basic_shares: Basic share count
        stock_price: Current stock price
        tax_rate: Tax rate

    Returns:
        Dictionary with different adjustment views
    """
    after_tax_sbc = stock_comp_expense * (1 - tax_rate)

    # View 1: Keep as expense (GAAP EPS)
    gaap_eps = reported_income / diluted_shares if diluted_shares > 0 else 0

    # View 2: Add back SBC (cash EPS)
    cash_earnings = reported_income + after_tax_sbc
    cash_eps = cash_earnings / basic_shares if basic_shares > 0 else 0

    # View 3: Economic dilution approach
    # Cost of SBC = dilution × stock price
    dilution_shares = diluted_shares - basic_shares
    dilution_cost = dilution_shares * stock_price
    economic_cost = min(after_tax_sbc, dilution_cost)  # True economic cost

    return {
        'reported_income': reported_income,
        'sbc_expense': stock_comp_expense,
        'after_tax_sbc': after_tax_sbc,
        'gaap_eps': gaap_eps,
        'cash_eps': cash_eps,
        'dilution_shares': dilution_shares,
        'dilution_percentage': dilution_shares / basic_shares if basic_shares > 0 else 0,
        'economic_sbc_cost': economic_cost,
    }


# =============================================================================
# LIFO to FIFO Conversion
# =============================================================================

def lifo_to_fifo_adjustment(
    lifo_inventory: float,
    lifo_reserve: float,
    prior_lifo_reserve: float,
    cogs_lifo: float,
    tax_rate: float = 0.25
) -> Dict[str, float]:
    """
    Convert LIFO to FIFO Inventory Method.

    LIFO Reserve = FIFO Inventory - LIFO Inventory

    Args:
        lifo_inventory: Inventory under LIFO
        lifo_reserve: LIFO reserve (from footnotes)
        prior_lifo_reserve: Prior year LIFO reserve
        cogs_lifo: COGS under LIFO
        tax_rate: Tax rate

    Returns:
        Dictionary with FIFO-equivalent values
    """
    # FIFO inventory
    fifo_inventory = lifo_inventory + lifo_reserve

    # FIFO COGS
    lifo_reserve_change = lifo_reserve - prior_lifo_reserve
    fifo_cogs = cogs_lifo - lifo_reserve_change

    # Income adjustment
    income_adjustment = lifo_reserve_change * (1 - tax_rate)

    # Equity adjustment (cumulative)
    equity_adjustment = lifo_reserve * (1 - tax_rate)

    return {
        'lifo_inventory': lifo_inventory,
        'fifo_inventory': fifo_inventory,
        'inventory_adjustment': lifo_reserve,
        'lifo_cogs': cogs_lifo,
        'fifo_cogs': fifo_cogs,
        'cogs_adjustment': -lifo_reserve_change,
        'income_adjustment': income_adjustment,
        'equity_adjustment': equity_adjustment,
    }


# =============================================================================
# Goodwill and Intangibles
# =============================================================================

def adjust_goodwill(
    total_assets: float,
    goodwill: float,
    intangible_assets: float,
    equity: float,
    net_income: float,
    approach: str = 'conservative'
) -> Dict[str, float]:
    """
    Adjust for Goodwill and Intangibles.

    Different approaches:
    - 'conservative': Remove all goodwill from equity
    - 'moderate': Keep identifiable intangibles, remove goodwill
    - 'aggressive': Keep all (standard GAAP)

    Args:
        total_assets: Total assets
        goodwill: Goodwill
        intangible_assets: Other intangible assets
        equity: Total equity
        net_income: Net income
        approach: Adjustment approach

    Returns:
        Dictionary with adjusted values
    """
    if approach == 'conservative':
        # Remove all intangibles
        assets_adj = total_assets - goodwill - intangible_assets
        equity_adj = equity - goodwill - intangible_assets
    elif approach == 'moderate':
        # Remove only goodwill
        assets_adj = total_assets - goodwill
        equity_adj = equity - goodwill
    else:
        # Keep all
        assets_adj = total_assets
        equity_adj = equity

    # Adjusted ROE
    roe_reported = net_income / equity if equity > 0 else 0
    roe_adjusted = net_income / equity_adj if equity_adj > 0 else 0

    # Tangible book value
    tangible_equity = equity - goodwill - intangible_assets

    return {
        'total_assets_reported': total_assets,
        'total_assets_adjusted': assets_adj,
        'goodwill': goodwill,
        'intangibles': intangible_assets,
        'equity_reported': equity,
        'equity_adjusted': equity_adj,
        'tangible_equity': tangible_equity,
        'roe_reported': roe_reported,
        'roe_adjusted': roe_adjusted,
        'approach': approach,
    }


# =============================================================================
# Pro-Forma Analysis Class
# =============================================================================

@dataclass
class ProFormaAdjustments:
    """Summary of all pro-forma adjustments."""
    lease_adjustment: Optional[LeaseCapitalizationResult]
    rd_adjustment: Optional[RDCapitalizationResult]
    pension_adjustment: Optional[PensionAdjustmentResult]
    normalized_earnings: Optional[NormalizedEarningsResult]
    total_debt_adjustment: float
    total_asset_adjustment: float
    total_equity_adjustment: float
    total_income_adjustment: float
    adjusted_roic: Optional[float]


class ProFormaAnalysis:
    """
    Comprehensive Pro-Forma Adjustment Analysis.

    Applies multiple adjustments to standardize financial statements.
    """

    def __init__(
        self,
        total_assets: float,
        total_debt: float,
        total_equity: float,
        net_income: float,
        operating_income: float,
        tax_rate: float = 0.25
    ):
        """Initialize pro-forma analysis."""
        self.total_assets = total_assets
        self.total_debt = total_debt
        self.total_equity = total_equity
        self.net_income = net_income
        self.operating_income = operating_income
        self.tax_rate = tax_rate

        self.adjustments: Dict = {}

    def add_lease_adjustment(
        self,
        annual_rent: float,
        remaining_term: float = 8.0,
        discount_rate: float = 0.05
    ) -> LeaseCapitalizationResult:
        """Add operating lease capitalization."""
        result = quick_lease_capitalization(
            annual_rent,
            discount_rate,
            remaining_term
        )
        self.adjustments['lease'] = result
        return result

    def add_rd_adjustment(
        self,
        rd_expenses: List[float],
        amortization_period: int = 5
    ) -> RDCapitalizationResult:
        """Add R&D capitalization."""
        result = capitalize_rd(
            rd_expenses,
            amortization_period,
            self.tax_rate
        )
        self.adjustments['rd'] = result
        return result

    def add_pension_adjustment(
        self,
        pbo: float,
        plan_assets: float,
        service_cost: float,
        interest_cost: float,
        expected_return: float,
        reported_expense: float
    ) -> PensionAdjustmentResult:
        """Add pension adjustment."""
        result = adjust_pension(
            pbo, plan_assets, service_cost,
            interest_cost, expected_return,
            reported_expense, self.tax_rate
        )
        self.adjustments['pension'] = result
        return result

    def add_earnings_normalization(
        self,
        restructuring: float = 0,
        impairments: float = 0,
        litigation: float = 0,
        asset_sales: float = 0,
        discontinued: float = 0,
        tax_adj: float = 0,
        other: float = 0
    ) -> NormalizedEarningsResult:
        """Add earnings normalization."""
        result = normalize_earnings(
            self.net_income,
            restructuring, impairments, litigation,
            asset_sales, discontinued, tax_adj, other,
            self.tax_rate
        )
        self.adjustments['normalize'] = result
        return result

    def get_adjusted_financials(self) -> ProFormaAdjustments:
        """
        Calculate total adjusted financials.

        Returns:
            ProFormaAdjustments with all adjustments
        """
        # Sum adjustments
        total_debt_adj = 0.0
        total_asset_adj = 0.0
        total_equity_adj = 0.0
        total_income_adj = 0.0

        lease_result = self.adjustments.get('lease')
        if lease_result:
            total_debt_adj += lease_result.debt_adjustment
            total_asset_adj += lease_result.asset_adjustment

        rd_result = self.adjustments.get('rd')
        if rd_result:
            total_asset_adj += rd_result.rd_asset
            total_equity_adj += rd_result.equity_adjustment
            total_income_adj += rd_result.net_income_adjustment

        pension_result = self.adjustments.get('pension')
        if pension_result:
            if pension_result.pension_asset_or_liability < 0:
                total_debt_adj += abs(pension_result.pension_asset_or_liability)

        normalize_result = self.adjustments.get('normalize')
        if normalize_result:
            total_income_adj += normalize_result.total_adjustments

        # Adjusted ROIC
        adjusted_nopat = self.operating_income * (1 - self.tax_rate) + total_income_adj
        adjusted_capital = self.total_debt + self.total_equity + total_debt_adj + total_equity_adj + total_asset_adj
        adjusted_roic = adjusted_nopat / adjusted_capital if adjusted_capital > 0 else None

        return ProFormaAdjustments(
            lease_adjustment=lease_result,
            rd_adjustment=rd_result,
            pension_adjustment=pension_result,
            normalized_earnings=normalize_result,
            total_debt_adjustment=total_debt_adj,
            total_asset_adjustment=total_asset_adj,
            total_equity_adjustment=total_equity_adj,
            total_income_adjustment=total_income_adj,
            adjusted_roic=adjusted_roic,
        )

    def get_adjusted_ratios(self) -> Dict[str, float]:
        """
        Calculate adjusted financial ratios.

        Returns:
            Dictionary of adjusted ratios
        """
        adj = self.get_adjusted_financials()

        adj_assets = self.total_assets + adj.total_asset_adjustment
        adj_debt = self.total_debt + adj.total_debt_adjustment
        adj_equity = self.total_equity + adj.total_equity_adjustment
        adj_income = self.net_income + adj.total_income_adjustment

        return {
            'adjusted_total_assets': adj_assets,
            'adjusted_total_debt': adj_debt,
            'adjusted_equity': adj_equity,
            'adjusted_net_income': adj_income,
            'adjusted_debt_to_equity': adj_debt / adj_equity if adj_equity > 0 else 0,
            'adjusted_debt_to_assets': adj_debt / adj_assets if adj_assets > 0 else 0,
            'adjusted_roe': adj_income / adj_equity if adj_equity > 0 else 0,
            'adjusted_roa': adj_income / adj_assets if adj_assets > 0 else 0,
            'adjusted_roic': adj.adjusted_roic,
        }
