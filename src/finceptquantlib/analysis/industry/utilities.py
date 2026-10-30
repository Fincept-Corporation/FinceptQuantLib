"""
Utilities & Regulated Industries Analysis

Specialized financial metrics for utilities and regulated industries:
- Rate Base and Allowed Return Analysis
- Regulatory Metrics
- Operational Efficiency (Heat Rate, Capacity Factor)
- Credit Metrics (FFO/Debt, Coverage Ratios)
- Growth and Capital Investment Analysis
"""

from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class UtilityFinancials:
    """Financial data structure for utilities and regulated industries."""

    # Income Statement
    total_revenues: float = 0.0
    fuel_costs: float = 0.0
    purchased_power: float = 0.0
    operation_maintenance: float = 0.0
    depreciation: float = 0.0
    taxes_other_than_income: float = 0.0
    interest_expense: float = 0.0
    income_tax_expense: float = 0.0
    net_income: float = 0.0

    # Rate Base Components (Regulatory)
    utility_plant_gross: float = 0.0
    accumulated_depreciation: float = 0.0
    construction_work_in_progress: float = 0.0
    working_capital_allowance: float = 0.0
    regulatory_assets: float = 0.0
    regulatory_liabilities: float = 0.0

    # Balance Sheet
    total_assets: float = 0.0
    total_debt: float = 0.0
    preferred_stock: float = 0.0
    common_equity: float = 0.0
    total_capitalization: float = 0.0

    # Cash Flow
    cash_from_operations: float = 0.0
    capital_expenditures: float = 0.0
    dividends_paid: float = 0.0

    # Regulatory Data
    allowed_roe: float = 0.0  # Allowed return on equity
    actual_earned_roe: float = 0.0
    authorized_rate_base: float = 0.0

    # Operational Data (Electric Utilities)
    net_generation_mwh: float = 0.0
    capacity_mw: float = 0.0
    peak_demand_mw: float = 0.0
    heat_rate_btu_kwh: float = 0.0  # Efficiency measure
    fuel_cost_per_mwh: float = 0.0

    # Customer Data
    number_of_customers: int = 0
    residential_customers: int = 0
    commercial_customers: int = 0
    industrial_customers: int = 0
    kwh_sales: float = 0.0

    # Prior Period (for growth)
    prior_year_customers: int = 0
    prior_year_rate_base: float = 0.0

    @property
    def operating_income(self) -> float:
        """Operating income."""
        return (self.total_revenues - self.fuel_costs - self.purchased_power -
                self.operation_maintenance - self.depreciation -
                self.taxes_other_than_income)

    @property
    def ebitda(self) -> float:
        """EBITDA."""
        return self.operating_income + self.depreciation

    @property
    def funds_from_operations(self) -> float:
        """FFO (simplified)."""
        return self.cash_from_operations


# =============================================================================
# Rate Base and Return Metrics
# =============================================================================

def rate_base(
    utility_plant: float,
    accumulated_depreciation: float,
    cwip: float = 0.0,
    working_capital: float = 0.0,
    regulatory_assets: float = 0.0,
    regulatory_liabilities: float = 0.0
) -> float:
    """
    Rate Base Calculation.

    Regulatory asset base on which utility earns a return.

    Rate Base = Net Plant + CWIP + Working Capital + Reg Assets - Reg Liabilities

    Args:
        utility_plant: Gross utility plant in service
        accumulated_depreciation: Accumulated depreciation
        cwip: Construction work in progress
        working_capital: Working capital allowance
        regulatory_assets: Deferred regulatory assets
        regulatory_liabilities: Deferred regulatory liabilities

    Returns:
        Rate base amount
    """
    net_plant = utility_plant - accumulated_depreciation
    return (net_plant + cwip + working_capital +
            regulatory_assets - regulatory_liabilities)


def allowed_return(
    rate_base_value: float,
    allowed_roe: float,
    cost_of_debt: float,
    equity_ratio: float
) -> float:
    """
    Allowed Return on Rate Base.

    Calculates allowed dollar return based on authorized ROE and capital structure.

    Allowed Return = Rate Base × WACC
    where WACC = (Equity Ratio × ROE) + (Debt Ratio × Cost of Debt)

    Args:
        rate_base_value: Authorized rate base
        allowed_roe: Authorized return on equity
        cost_of_debt: Embedded cost of debt
        equity_ratio: Equity as percent of capitalization

    Returns:
        Allowed dollar return
    """
    debt_ratio = 1.0 - equity_ratio
    wacc = (equity_ratio * allowed_roe) + (debt_ratio * cost_of_debt)
    return rate_base_value * wacc


def earned_return_on_rate_base(
    operating_income: float,
    rate_base_value: float
) -> Optional[float]:
    """
    Earned Return on Rate Base.

    Actual return earned vs. rate base.

    Args:
        operating_income: Utility operating income
        rate_base_value: Rate base

    Returns:
        Earned return as decimal
    """
    if rate_base_value <= 0:
        return None
    return operating_income / rate_base_value


def regulatory_lag(
    allowed_roe: float,
    earned_roe: float
) -> float:
    """
    Regulatory Lag.

    Gap between authorized and earned return.

    Positive = under-earning
    Negative = over-earning

    Args:
        allowed_roe: Authorized return on equity
        earned_roe: Actually earned return on equity

    Returns:
        Lag as decimal (positive = under-earning)
    """
    return allowed_roe - earned_roe


def rate_case_outcome(
    requested_increase: float,
    granted_increase: float
) -> Optional[float]:
    """
    Rate Case Outcome.

    Percentage of requested rate increase granted.

    Args:
        requested_increase: Revenue increase requested
        granted_increase: Revenue increase granted

    Returns:
        Outcome ratio as decimal
    """
    if requested_increase <= 0:
        return None
    return granted_increase / requested_increase


# =============================================================================
# Operational Efficiency Metrics
# =============================================================================

def capacity_factor(
    actual_generation_mwh: float,
    capacity_mw: float,
    hours_in_period: float = 8760.0
) -> Optional[float]:
    """
    Capacity Factor.

    Actual generation as percentage of theoretical maximum.

    Capacity Factor = Actual Generation / (Capacity × Hours)

    Typical ranges:
    - Nuclear: 90%+
    - Natural Gas Combined Cycle: 40-60%
    - Wind: 25-40%
    - Solar: 15-25%

    Args:
        actual_generation_mwh: Actual MWh generated
        capacity_mw: Nameplate capacity in MW
        hours_in_period: Hours in period (8760 for full year)

    Returns:
        Capacity factor as decimal
    """
    max_generation = capacity_mw * hours_in_period
    if max_generation <= 0:
        return None
    return actual_generation_mwh / max_generation


def heat_rate(
    fuel_consumed_btu: float,
    net_generation_kwh: float
) -> Optional[float]:
    """
    Heat Rate (Thermal Efficiency).

    BTUs of fuel per kWh generated. Lower is more efficient.

    Heat Rate = Fuel BTU / Net Generation kWh

    Typical ranges:
    - Efficient CCGT: 6,000-7,000 BTU/kWh
    - Coal: 9,000-11,000 BTU/kWh
    - Older Gas: 10,000-12,000 BTU/kWh

    Args:
        fuel_consumed_btu: Total fuel consumed in BTU
        net_generation_kwh: Net generation in kWh

    Returns:
        Heat rate in BTU/kWh
    """
    if net_generation_kwh <= 0:
        return None
    return fuel_consumed_btu / net_generation_kwh


def reserve_margin(
    installed_capacity: float,
    peak_demand: float
) -> Optional[float]:
    """
    Reserve Margin.

    Excess capacity above peak demand.

    Reserve Margin = (Capacity - Peak Demand) / Peak Demand

    Typical target: 15-20%

    Args:
        installed_capacity: Total installed generating capacity
        peak_demand: Peak system demand

    Returns:
        Reserve margin as decimal
    """
    if peak_demand <= 0:
        return None
    return (installed_capacity - peak_demand) / peak_demand


def system_load_factor(
    total_energy_mwh: float,
    peak_demand_mw: float,
    hours_in_period: float = 8760.0
) -> Optional[float]:
    """
    System Load Factor.

    Average load as percentage of peak load.

    Load Factor = Average Load / Peak Load
                = Total Energy / (Peak × Hours)

    Higher = more efficient utilization of generating capacity.

    Args:
        total_energy_mwh: Total energy delivered in MWh
        peak_demand_mw: Peak demand in MW
        hours_in_period: Hours in period

    Returns:
        Load factor as decimal
    """
    if peak_demand_mw <= 0 or hours_in_period <= 0:
        return None
    max_energy = peak_demand_mw * hours_in_period
    return total_energy_mwh / max_energy


def fuel_cost_recovery(
    fuel_costs_collected: float,
    actual_fuel_costs: float
) -> Optional[float]:
    """
    Fuel Cost Recovery Rate.

    Measures adequacy of fuel clause recovery.

    Recovery = Collected / Actual

    < 1.0 = under-recovery (margin compression)
    > 1.0 = over-recovery (to be refunded)

    Args:
        fuel_costs_collected: Fuel costs recovered through rates
        actual_fuel_costs: Actual fuel costs incurred

    Returns:
        Recovery ratio
    """
    if actual_fuel_costs <= 0:
        return None
    return fuel_costs_collected / actual_fuel_costs


# =============================================================================
# Customer and Revenue Metrics
# =============================================================================

def customer_growth_rate(
    current_customers: int,
    prior_customers: int
) -> Optional[float]:
    """
    Customer Growth Rate.

    Year-over-year customer growth.

    Args:
        current_customers: Current period customer count
        prior_customers: Prior period customer count

    Returns:
        Growth rate as decimal
    """
    if prior_customers <= 0:
        return None
    return (current_customers - prior_customers) / prior_customers


def revenue_per_customer(
    total_revenues: float,
    number_of_customers: int
) -> Optional[float]:
    """
    Revenue per Customer.

    Average revenue per customer.

    Args:
        total_revenues: Total utility revenues
        number_of_customers: Total customers

    Returns:
        Revenue per customer
    """
    if number_of_customers <= 0:
        return None
    return total_revenues / number_of_customers


def revenue_per_kwh(
    total_revenues: float,
    kwh_sales: float
) -> Optional[float]:
    """
    Revenue per kWh.

    Average realized price.

    Args:
        total_revenues: Total utility revenues
        kwh_sales: Total kWh sold

    Returns:
        Revenue per kWh ($/kWh)
    """
    if kwh_sales <= 0:
        return None
    return total_revenues / kwh_sales


def customer_mix_analysis(
    residential_customers: int,
    commercial_customers: int,
    industrial_customers: int
) -> Dict[str, float]:
    """
    Customer Mix Analysis.

    Distribution of customers by class.

    Args:
        residential_customers: Residential customer count
        commercial_customers: Commercial customer count
        industrial_customers: Industrial customer count

    Returns:
        Dictionary with customer mix percentages
    """
    total = residential_customers + commercial_customers + industrial_customers
    if total <= 0:
        return {'residential': 0, 'commercial': 0, 'industrial': 0}

    return {
        'residential': residential_customers / total,
        'commercial': commercial_customers / total,
        'industrial': industrial_customers / total,
    }


# =============================================================================
# Credit and Financial Metrics
# =============================================================================

def operating_ratio_utility(
    operating_expenses: float,
    operating_revenues: float
) -> Optional[float]:
    """
    Operating Ratio.

    Operating expenses as percentage of revenues.
    Lower is more efficient.

    Args:
        operating_expenses: Total operating expenses
        operating_revenues: Total operating revenues

    Returns:
        Ratio as decimal
    """
    if operating_revenues <= 0:
        return None
    return operating_expenses / operating_revenues


def debt_to_capitalization(
    total_debt: float,
    total_capitalization: float
) -> Optional[float]:
    """
    Debt to Capitalization.

    Capital structure leverage.

    Typical utility range: 45-55%

    Args:
        total_debt: Total debt (short + long term)
        total_capitalization: Total capital (debt + preferred + equity)

    Returns:
        Ratio as decimal
    """
    if total_capitalization <= 0:
        return None
    return total_debt / total_capitalization


def times_interest_earned_utility(
    operating_income: float,
    interest_expense: float
) -> Optional[float]:
    """
    Times Interest Earned.

    Interest coverage for utilities.

    Typical minimum: 2.5x

    Args:
        operating_income: Operating income
        interest_expense: Interest expense

    Returns:
        Coverage ratio
    """
    if interest_expense <= 0:
        return None
    return operating_income / interest_expense


def ffo_to_debt(
    funds_from_operations: float,
    total_debt: float
) -> Optional[float]:
    """
    FFO to Debt.

    Key credit metric for utilities.

    Rating thresholds (S&P):
    - A: > 20%
    - BBB: 13-20%
    - BB: 9-13%

    Args:
        funds_from_operations: Cash from operations (adjusted)
        total_debt: Total debt

    Returns:
        Ratio as decimal
    """
    if total_debt <= 0:
        return None
    return funds_from_operations / total_debt


def ffo_interest_coverage(
    ffo: float,
    interest_expense: float
) -> Optional[float]:
    """
    FFO Interest Coverage.

    Cash-based interest coverage.

    Typical minimum: 4.0x

    Args:
        ffo: Funds from operations
        interest_expense: Interest expense

    Returns:
        Coverage ratio
    """
    if interest_expense <= 0:
        return None
    return ffo / interest_expense


def retained_cash_flow_to_capex(
    cfo: float,
    dividends: float,
    capex: float
) -> Optional[float]:
    """
    Retained Cash Flow to CapEx.

    Internal funding capability.

    > 1.0 = can fund capex internally
    < 1.0 = requires external financing

    Args:
        cfo: Cash from operations
        dividends: Dividends paid
        capex: Capital expenditures

    Returns:
        Ratio
    """
    if capex <= 0:
        return None
    retained = cfo - dividends
    return retained / capex


def capex_to_depreciation(
    capital_expenditures: float,
    depreciation: float
) -> Optional[float]:
    """
    CapEx to Depreciation.

    Capital reinvestment rate.

    > 1.0 = growing asset base
    < 1.0 = declining asset base

    Typical utility: 1.5x - 2.5x (growth period)

    Args:
        capital_expenditures: Annual capex
        depreciation: Annual depreciation

    Returns:
        Ratio
    """
    if depreciation <= 0:
        return None
    return capital_expenditures / depreciation


def rate_base_growth(
    current_rate_base: float,
    prior_rate_base: float
) -> Optional[float]:
    """
    Rate Base Growth Rate.

    Regulatory asset growth.

    Args:
        current_rate_base: Current rate base
        prior_rate_base: Prior year rate base

    Returns:
        Growth rate as decimal
    """
    if prior_rate_base <= 0:
        return None
    return (current_rate_base - prior_rate_base) / prior_rate_base


# =============================================================================
# Utilities Analysis Class
# =============================================================================

@dataclass
class RegulatoryMetrics:
    """Regulatory analysis summary."""
    rate_base: float
    earned_return: Optional[float]
    allowed_return: Optional[float]
    regulatory_lag: Optional[float]
    assessment: str = ""


@dataclass
class OperationalMetrics:
    """Operational efficiency summary."""
    capacity_factor: Optional[float]
    heat_rate: Optional[float]
    reserve_margin: Optional[float]
    load_factor: Optional[float]
    assessment: str = ""


@dataclass
class CreditMetrics:
    """Credit analysis summary."""
    ffo_to_debt: Optional[float]
    interest_coverage: Optional[float]
    debt_to_cap: Optional[float]
    implied_rating: str = ""


class UtilityAnalysis:
    """
    Comprehensive utility analysis.

    Provides specialized analysis for electric, gas, and water utilities.
    """

    def __init__(self, financials: UtilityFinancials):
        """Initialize with utility financials."""
        self.financials = financials

    def analyze_regulatory(self) -> RegulatoryMetrics:
        """
        Analyze regulatory metrics.

        Returns:
            Regulatory metrics with assessment
        """
        f = self.financials

        rb = rate_base(
            f.utility_plant_gross,
            f.accumulated_depreciation,
            f.construction_work_in_progress,
            f.working_capital_allowance,
            f.regulatory_assets,
            f.regulatory_liabilities
        )

        earned = earned_return_on_rate_base(f.operating_income, rb) if rb > 0 else None
        lag = regulatory_lag(f.allowed_roe, f.actual_earned_roe)

        # Assessment
        assessment_parts = []
        if lag > 0.02:
            assessment_parts.append(f"Significant under-earning ({lag*100:.1f}% below allowed)")
        elif lag > 0:
            assessment_parts.append("Slight under-earning")
        elif lag < -0.01:
            assessment_parts.append("Over-earning (regulatory risk)")
        else:
            assessment_parts.append("Earning close to allowed return")

        return RegulatoryMetrics(
            rate_base=rb,
            earned_return=earned,
            allowed_return=f.allowed_roe,
            regulatory_lag=lag,
            assessment=" | ".join(assessment_parts)
        )

    def analyze_operations(self) -> OperationalMetrics:
        """
        Analyze operational efficiency.

        Returns:
            Operational metrics with assessment
        """
        f = self.financials

        cf = capacity_factor(f.net_generation_mwh, f.capacity_mw)
        hr = f.heat_rate_btu_kwh if f.heat_rate_btu_kwh > 0 else None
        rm = reserve_margin(f.capacity_mw, f.peak_demand_mw)
        lf = system_load_factor(f.net_generation_mwh, f.peak_demand_mw)

        # Assessment
        assessment_parts = []

        if cf is not None:
            if cf > 0.85:
                assessment_parts.append("High capacity utilization")
            elif cf < 0.40:
                assessment_parts.append("Low capacity utilization")

        if hr is not None:
            if hr < 7000:
                assessment_parts.append("Efficient heat rate")
            elif hr > 10000:
                assessment_parts.append("Inefficient heat rate")

        if rm is not None:
            if rm < 0.10:
                assessment_parts.append("Low reserve margin (reliability risk)")
            elif rm > 0.25:
                assessment_parts.append("High reserve margin")

        return OperationalMetrics(
            capacity_factor=cf,
            heat_rate=hr,
            reserve_margin=rm,
            load_factor=lf,
            assessment=" | ".join(assessment_parts) if assessment_parts else "Normal operations"
        )

    def analyze_credit(self) -> CreditMetrics:
        """
        Analyze credit metrics.

        Returns:
            Credit metrics with implied rating
        """
        f = self.financials

        ffo_debt = ffo_to_debt(f.funds_from_operations, f.total_debt)
        tie = times_interest_earned_utility(f.operating_income, f.interest_expense)
        d_cap = debt_to_capitalization(f.total_debt, f.total_capitalization)

        # Implied rating
        if ffo_debt is not None:
            if ffo_debt > 0.20:
                rating = "A category"
            elif ffo_debt > 0.13:
                rating = "BBB category"
            elif ffo_debt > 0.09:
                rating = "BB category"
            else:
                rating = "Below investment grade"
        else:
            rating = "Unable to determine"

        return CreditMetrics(
            ffo_to_debt=ffo_debt,
            interest_coverage=tie,
            debt_to_cap=d_cap,
            implied_rating=rating
        )

    def analyze_growth(self) -> Dict[str, Optional[float]]:
        """
        Analyze growth metrics.

        Returns:
            Dictionary of growth metrics
        """
        f = self.financials

        return {
            'customer_growth': customer_growth_rate(
                f.number_of_customers,
                f.prior_year_customers
            ),
            'rate_base_growth': rate_base_growth(
                rate_base(
                    f.utility_plant_gross,
                    f.accumulated_depreciation,
                    f.construction_work_in_progress,
                    f.working_capital_allowance,
                    f.regulatory_assets,
                    f.regulatory_liabilities
                ),
                f.prior_year_rate_base
            ),
            'capex_to_depreciation': capex_to_depreciation(
                f.capital_expenditures,
                f.depreciation
            ),
            'internal_funding': retained_cash_flow_to_capex(
                f.cash_from_operations,
                f.dividends_paid,
                f.capital_expenditures
            ),
        }

    def analyze_customer_metrics(self) -> Dict[str, Optional[float]]:
        """
        Analyze customer and revenue metrics.

        Returns:
            Dictionary of customer metrics
        """
        f = self.financials

        mix = customer_mix_analysis(
            f.residential_customers,
            f.commercial_customers,
            f.industrial_customers
        )

        return {
            'revenue_per_customer': revenue_per_customer(
                f.total_revenues,
                f.number_of_customers
            ),
            'revenue_per_kwh': revenue_per_kwh(
                f.total_revenues,
                f.kwh_sales
            ),
            'customer_mix_residential': mix['residential'],
            'customer_mix_commercial': mix['commercial'],
            'customer_mix_industrial': mix['industrial'],
        }

    def full_analysis(self) -> Dict:
        """
        Perform complete utility analysis.

        Returns:
            Dictionary with all analysis components
        """
        return {
            'regulatory': self.analyze_regulatory(),
            'operations': self.analyze_operations(),
            'credit': self.analyze_credit(),
            'growth': self.analyze_growth(),
            'customer_metrics': self.analyze_customer_metrics(),
        }
