"""
Capital Structure Optimization Module

Analysis and optimization of firm capital structure:
- Trade-Off Theory (Tax Shield vs. Distress Costs)
- WACC Minimization
- Optimal Debt Ratio Calculation
- Tax Shield Valuation
- Financial Distress Costs
- Pecking Order Analysis
- Capital Structure Ratios and Benchmarks
"""

from dataclasses import dataclass
from typing import Dict, List
import math


# =============================================================================
# Tax Shield Valuation
# =============================================================================

def tax_shield_value(
    debt: float,
    tax_rate: float,
    perpetual: bool = True,
    debt_maturity: float = 10.0,
    risk_free_rate: float = 0.05
) -> float:
    """
    Present Value of Tax Shield.

    For perpetual debt (MM with taxes):
    PV(Tax Shield) = Debt × Tax Rate

    For finite debt:
    PV(Tax Shield) = Σ(Interest × Tax Rate) / (1+r)^t

    Args:
        debt: Amount of debt
        tax_rate: Corporate tax rate
        perpetual: Whether debt is perpetual
        debt_maturity: Average debt maturity (if not perpetual)
        risk_free_rate: Discount rate for tax shield

    Returns:
        Present value of tax shield
    """
    if perpetual:
        # Modigliani-Miller with taxes
        return debt * tax_rate
    else:
        # Finite tax shield
        # Assuming constant debt and interest
        annual_tax_shield = debt * risk_free_rate * tax_rate

        # Present value of annuity
        if risk_free_rate > 0:
            pv_factor = (1 - (1 + risk_free_rate) ** (-debt_maturity)) / risk_free_rate
            return annual_tax_shield * pv_factor
        else:
            return annual_tax_shield * debt_maturity


def marginal_tax_shield(
    additional_debt: float,
    tax_rate: float,
    interest_rate: float
) -> float:
    """
    Marginal Tax Shield from Additional Debt.

    Annual tax savings from incremental debt.

    Marginal Tax Shield = ΔDebt × Interest Rate × Tax Rate

    Args:
        additional_debt: Additional debt amount
        tax_rate: Marginal tax rate
        interest_rate: Interest rate on new debt

    Returns:
        Annual tax shield from new debt
    """
    return additional_debt * interest_rate * tax_rate


# =============================================================================
# Financial Distress Costs
# =============================================================================

def expected_distress_cost(
    probability_of_distress: float,
    distress_cost_percentage: float,
    firm_value: float
) -> float:
    """
    Expected Cost of Financial Distress.

    E(Distress Cost) = P(Distress) × Cost if Distressed × Firm Value

    Args:
        probability_of_distress: Probability of financial distress
        distress_cost_percentage: Distress costs as % of firm value (typically 10-25%)
        firm_value: Firm value

    Returns:
        Expected distress cost
    """
    return probability_of_distress * distress_cost_percentage * firm_value


def distress_probability_from_leverage(
    leverage: float,
    volatility: float,
    base_probability: float = 0.02
) -> float:
    """
    Estimate Distress Probability from Leverage.

    Simplified model where distress probability increases with leverage.

    P(Distress) = base × exp(α × leverage × volatility)

    Args:
        leverage: Debt / Total Capital
        volatility: Asset or earnings volatility
        base_probability: Base distress probability at zero leverage

    Returns:
        Estimated probability of distress
    """
    # Exponential increase with leverage and volatility
    alpha = 5.0  # Sensitivity parameter
    prob = base_probability * math.exp(alpha * leverage * volatility)
    return min(prob, 1.0)


def indirect_distress_costs(
    revenue: float,
    distress_revenue_impact: float = 0.10,
    operating_margin: float = 0.10
) -> float:
    """
    Indirect Costs of Financial Distress.

    Includes:
    - Lost sales from customer concerns
    - Supplier unwillingness to extend credit
    - Key employee departures
    - Reduced investment in positive NPV projects

    Args:
        revenue: Annual revenue
        distress_revenue_impact: Revenue decline in distress
        operating_margin: Operating profit margin

    Returns:
        Indirect distress cost (lost profits)
    """
    lost_revenue = revenue * distress_revenue_impact
    lost_profit = lost_revenue * operating_margin
    return lost_profit


def direct_distress_costs(
    firm_value: float,
    legal_costs_pct: float = 0.03,
    administrative_costs_pct: float = 0.02,
    asset_fire_sale_discount: float = 0.20,
    asset_fraction_sold: float = 0.30
) -> float:
    """
    Direct Costs of Financial Distress.

    Includes:
    - Legal and administrative costs
    - Asset fire-sale discounts

    Args:
        firm_value: Firm value
        legal_costs_pct: Legal costs as % of firm value
        administrative_costs_pct: Admin costs as % of firm value
        asset_fire_sale_discount: Discount on asset sales
        asset_fraction_sold: Fraction of assets sold

    Returns:
        Direct distress costs
    """
    legal_admin = firm_value * (legal_costs_pct + administrative_costs_pct)
    fire_sale_loss = firm_value * asset_fraction_sold * asset_fire_sale_discount
    return legal_admin + fire_sale_loss


# =============================================================================
# Optimal Capital Structure
# =============================================================================

def optimal_debt_ratio_tradeoff(
    unlevered_value: float,
    tax_rate: float,
    cost_of_debt: float,
    asset_volatility: float,
    distress_cost_pct: float = 0.20,
    steps: int = 100
) -> Dict[str, float]:
    """
    Find Optimal Debt Ratio using Trade-Off Theory.

    Maximizes: V_L = V_U + PV(Tax Shield) - PV(Distress Costs)

    Args:
        unlevered_value: Unlevered firm value
        tax_rate: Corporate tax rate
        cost_of_debt: Cost of debt
        asset_volatility: Asset volatility (for distress probability)
        distress_cost_pct: Distress costs as % of firm value
        steps: Number of leverage ratios to evaluate

    Returns:
        Dictionary with optimal leverage and value components
    """
    best_leverage = 0
    best_value = unlevered_value
    results = []

    for i in range(steps + 1):
        leverage = i / steps

        if leverage >= 0.99:
            continue

        # Debt amount
        # Assuming levered value ≈ unlevered value for calculation
        debt = unlevered_value * leverage / (1 - leverage * tax_rate)

        # Tax shield
        ts = tax_shield_value(debt, tax_rate, perpetual=True)

        # Distress probability
        prob_distress = distress_probability_from_leverage(
            leverage,
            asset_volatility
        )

        # Expected distress cost
        levered_value_approx = unlevered_value + ts
        distress_cost = expected_distress_cost(
            prob_distress,
            distress_cost_pct,
            levered_value_approx
        )

        # Levered firm value
        levered_value = unlevered_value + ts - distress_cost

        results.append({
            'leverage': leverage,
            'debt': debt,
            'tax_shield': ts,
            'prob_distress': prob_distress,
            'distress_cost': distress_cost,
            'levered_value': levered_value,
        })

        if levered_value > best_value:
            best_value = levered_value
            best_leverage = leverage

    # Find the optimal result
    optimal_result = None
    for r in results:
        if abs(r['leverage'] - best_leverage) < 0.01:
            optimal_result = r
            break

    return {
        'optimal_leverage': best_leverage,
        'optimal_debt_ratio': best_leverage,
        'max_firm_value': best_value,
        'value_gain': best_value - unlevered_value,
        'optimal_tax_shield': optimal_result['tax_shield'] if optimal_result else 0,
        'optimal_distress_cost': optimal_result['distress_cost'] if optimal_result else 0,
        'all_results': results,
    }


def wacc_minimization(
    cost_of_equity_unlevered: float,
    cost_of_debt: float,
    tax_rate: float,
    equity_risk_premium_per_leverage: float = 0.02,
    debt_spread_per_leverage: float = 0.01,
    steps: int = 100
) -> Dict[str, float]:
    """
    Find Optimal Capital Structure by Minimizing WACC.

    As leverage increases:
    - Cost of equity increases (due to financial risk)
    - Cost of debt increases (due to credit risk)
    - Tax shield benefit increases

    WACC = (E/V) × Re + (D/V) × Rd × (1-T)

    Args:
        cost_of_equity_unlevered: Cost of equity at zero leverage
        cost_of_debt: Base cost of debt
        tax_rate: Corporate tax rate
        equity_risk_premium_per_leverage: Additional equity return per unit leverage
        debt_spread_per_leverage: Additional debt spread per unit leverage
        steps: Number of leverage ratios to evaluate

    Returns:
        Dictionary with optimal leverage and WACC
    """
    best_leverage = 0
    min_wacc = cost_of_equity_unlevered
    results = []

    for i in range(steps + 1):
        leverage = i / steps

        if leverage >= 0.95:
            continue

        # Cost of equity increases with leverage (Hamada equation style)
        equity_premium = equity_risk_premium_per_leverage * leverage / (1 - leverage)
        cost_of_equity = cost_of_equity_unlevered + equity_premium

        # Cost of debt increases with leverage
        debt_premium = debt_spread_per_leverage * leverage / (1 - leverage)
        cost_of_debt_levered = cost_of_debt + debt_premium

        # WACC
        equity_weight = 1 - leverage
        debt_weight = leverage
        wacc = (equity_weight * cost_of_equity +
                debt_weight * cost_of_debt_levered * (1 - tax_rate))

        results.append({
            'leverage': leverage,
            'cost_of_equity': cost_of_equity,
            'cost_of_debt': cost_of_debt_levered,
            'wacc': wacc,
        })

        if wacc < min_wacc:
            min_wacc = wacc
            best_leverage = leverage

    return {
        'optimal_leverage': best_leverage,
        'minimum_wacc': min_wacc,
        'wacc_at_zero_leverage': cost_of_equity_unlevered,
        'wacc_reduction': cost_of_equity_unlevered - min_wacc,
        'all_results': results,
    }


# =============================================================================
# Modigliani-Miller Propositions
# =============================================================================

def mm_proposition_1_no_taxes(
    unlevered_value: float,
    debt: float,
    equity: float
) -> Dict[str, float]:
    """
    Modigliani-Miller Proposition 1 (No Taxes).

    V_L = V_U

    Capital structure is irrelevant in perfect markets.

    Args:
        unlevered_value: Value of unlevered firm
        debt: Debt value
        equity: Equity value

    Returns:
        Dictionary comparing levered vs unlevered value
    """
    levered_value = debt + equity

    return {
        'unlevered_value': unlevered_value,
        'levered_value': levered_value,
        'difference': levered_value - unlevered_value,
        'mm1_holds': abs(levered_value - unlevered_value) < unlevered_value * 0.01,
    }


def mm_proposition_1_with_taxes(
    unlevered_value: float,
    debt: float,
    tax_rate: float
) -> Dict[str, float]:
    """
    Modigliani-Miller Proposition 1 (With Taxes).

    V_L = V_U + Tax Rate × Debt

    Levered firm value increases due to tax shield.

    Args:
        unlevered_value: Value of unlevered firm
        debt: Debt value
        tax_rate: Corporate tax rate

    Returns:
        Dictionary with levered value calculation
    """
    tax_shield = debt * tax_rate
    levered_value = unlevered_value + tax_shield

    return {
        'unlevered_value': unlevered_value,
        'tax_shield': tax_shield,
        'levered_value': levered_value,
        'value_increase_pct': tax_shield / unlevered_value if unlevered_value > 0 else 0,
    }


def mm_proposition_2(
    cost_of_equity_unlevered: float,
    cost_of_debt: float,
    debt: float,
    equity: float,
    tax_rate: float = 0.0
) -> float:
    """
    Modigliani-Miller Proposition 2.

    Cost of equity increases with leverage.

    Without taxes:
    Re = Ru + (Ru - Rd) × (D/E)

    With taxes:
    Re = Ru + (Ru - Rd) × (D/E) × (1 - T)

    Args:
        cost_of_equity_unlevered: Unlevered cost of equity
        cost_of_debt: Cost of debt
        debt: Debt value
        equity: Equity value
        tax_rate: Corporate tax rate

    Returns:
        Levered cost of equity
    """
    if equity <= 0:
        return cost_of_equity_unlevered

    de_ratio = debt / equity
    spread = cost_of_equity_unlevered - cost_of_debt

    return cost_of_equity_unlevered + spread * de_ratio * (1 - tax_rate)


def hamada_equation(
    beta_unlevered: float,
    debt: float,
    equity: float,
    tax_rate: float
) -> float:
    """
    Hamada Equation.

    Relationship between levered and unlevered beta.

    β_L = β_U × [1 + (1-T) × (D/E)]

    Args:
        beta_unlevered: Unlevered (asset) beta
        debt: Debt value
        equity: Equity value
        tax_rate: Corporate tax rate

    Returns:
        Levered equity beta
    """
    if equity <= 0:
        return beta_unlevered

    de_ratio = debt / equity
    return beta_unlevered * (1 + (1 - tax_rate) * de_ratio)


def unlever_beta(
    beta_levered: float,
    debt: float,
    equity: float,
    tax_rate: float
) -> float:
    """
    Unlever Beta (Reverse Hamada).

    β_U = β_L / [1 + (1-T) × (D/E)]

    Args:
        beta_levered: Levered equity beta
        debt: Debt value
        equity: Equity value
        tax_rate: Corporate tax rate

    Returns:
        Unlevered (asset) beta
    """
    if equity <= 0:
        return beta_levered

    de_ratio = debt / equity
    return beta_levered / (1 + (1 - tax_rate) * de_ratio)


# =============================================================================
# Capital Structure Ratios
# =============================================================================

def debt_capacity(
    ebitda: float,
    max_debt_to_ebitda: float = 4.0,
    interest_rate: float = 0.06,
    min_interest_coverage: float = 3.0
) -> Dict[str, float]:
    """
    Estimate Debt Capacity.

    Based on common lending covenants.

    Args:
        ebitda: Annual EBITDA
        max_debt_to_ebitda: Maximum debt/EBITDA ratio
        interest_rate: Expected interest rate
        min_interest_coverage: Minimum interest coverage ratio

    Returns:
        Dictionary with debt capacity estimates
    """
    # From debt/EBITDA constraint
    capacity_from_ebitda = ebitda * max_debt_to_ebitda

    # From interest coverage constraint
    max_interest = ebitda / min_interest_coverage
    capacity_from_coverage = max_interest / interest_rate if interest_rate > 0 else float('inf')

    # Binding constraint
    debt_capacity_amount = min(capacity_from_ebitda, capacity_from_coverage)

    return {
        'debt_capacity': debt_capacity_amount,
        'capacity_from_ebitda_covenant': capacity_from_ebitda,
        'capacity_from_coverage_covenant': capacity_from_coverage,
        'binding_constraint': 'ebitda' if capacity_from_ebitda < capacity_from_coverage else 'coverage',
        'max_debt_to_ebitda': max_debt_to_ebitda,
        'implied_interest_coverage': ebitda / (debt_capacity_amount * interest_rate) if debt_capacity_amount > 0 and interest_rate > 0 else float('inf'),
    }


def target_capital_structure(
    industry_median_leverage: float,
    current_leverage: float,
    credit_rating_target: str,
    profitability_adjustment: float = 0.0
) -> Dict[str, float]:
    """
    Determine Target Capital Structure.

    Considers:
    - Industry norms
    - Credit rating targets
    - Profitability (high profit → more debt capacity)

    Args:
        industry_median_leverage: Industry median D/V ratio
        current_leverage: Current D/V ratio
        credit_rating_target: Target credit rating
        profitability_adjustment: Adjustment for above/below average profitability

    Returns:
        Target capital structure recommendation
    """
    # Rating-implied leverage ranges
    rating_leverage = {
        'AAA': (0.0, 0.15),
        'AA': (0.10, 0.25),
        'A': (0.20, 0.35),
        'BBB': (0.30, 0.45),
        'BB': (0.40, 0.55),
        'B': (0.50, 0.65),
    }

    # Get range for target rating
    lev_range = rating_leverage.get(credit_rating_target.upper()[:3], (0.30, 0.45))

    # Target is midpoint of range, adjusted for profitability
    base_target = (lev_range[0] + lev_range[1]) / 2
    target = base_target + profitability_adjustment

    # Consider industry median
    industry_adjusted_target = (target + industry_median_leverage) / 2

    # Ensure within rating range
    final_target = max(lev_range[0], min(lev_range[1], industry_adjusted_target))

    return {
        'target_leverage': final_target,
        'current_leverage': current_leverage,
        'adjustment_needed': final_target - current_leverage,
        'rating_range': lev_range,
        'industry_median': industry_median_leverage,
        'recommendation': 'increase_debt' if final_target > current_leverage else 'reduce_debt',
    }


# =============================================================================
# Capital Structure Analysis Class
# =============================================================================

@dataclass
class CapitalStructureResult:
    """Capital structure analysis result."""
    optimal_leverage: float
    current_leverage: float
    tax_shield_value: float
    expected_distress_cost: float
    net_benefit: float
    wacc_current: float
    wacc_optimal: float
    recommendation: str


class CapitalStructureAnalysis:
    """
    Comprehensive Capital Structure Analysis.

    Combines trade-off theory, WACC minimization, and practical constraints.
    """

    def __init__(
        self,
        equity_value: float,
        debt: float,
        ebitda: float,
        tax_rate: float,
        cost_of_equity: float,
        cost_of_debt: float,
        asset_volatility: float = 0.30
    ):
        """Initialize capital structure analysis."""
        self.equity = equity_value
        self.debt = debt
        self.ebitda = ebitda
        self.tax_rate = tax_rate
        self.cost_of_equity = cost_of_equity
        self.cost_of_debt = cost_of_debt
        self.asset_volatility = asset_volatility

    @property
    def firm_value(self) -> float:
        """Total firm value."""
        return self.equity + self.debt

    @property
    def current_leverage(self) -> float:
        """Current debt-to-value ratio."""
        if self.firm_value <= 0:
            return 0
        return self.debt / self.firm_value

    @property
    def current_wacc(self) -> float:
        """Current WACC."""
        if self.firm_value <= 0:
            return self.cost_of_equity

        equity_weight = self.equity / self.firm_value
        debt_weight = self.debt / self.firm_value

        return (equity_weight * self.cost_of_equity +
                debt_weight * self.cost_of_debt * (1 - self.tax_rate))

    def analyze_trade_off(self) -> Dict:
        """
        Analyze optimal structure using trade-off theory.

        Returns:
            Trade-off analysis results
        """
        # Estimate unlevered value
        unlevered_value = self.firm_value - tax_shield_value(self.debt, self.tax_rate)

        return optimal_debt_ratio_tradeoff(
            unlevered_value,
            self.tax_rate,
            self.cost_of_debt,
            self.asset_volatility
        )

    def analyze_wacc(self) -> Dict:
        """
        Analyze optimal structure by minimizing WACC.

        Returns:
            WACC minimization results
        """
        # Estimate unlevered cost of equity
        unlevered_cost = unlever_beta(
            self.cost_of_equity - 0.05,  # Approximate beta from cost
            self.debt,
            self.equity,
            self.tax_rate
        ) + 0.05  # Add back risk-free rate

        return wacc_minimization(
            unlevered_cost,
            self.cost_of_debt,
            self.tax_rate
        )

    def debt_capacity_analysis(self) -> Dict:
        """
        Analyze debt capacity.

        Returns:
            Debt capacity analysis
        """
        return debt_capacity(
            self.ebitda,
            max_debt_to_ebitda=4.0,
            interest_rate=self.cost_of_debt
        )

    def full_analysis(self) -> CapitalStructureResult:
        """
        Perform comprehensive capital structure analysis.

        Returns:
            Complete analysis result
        """
        trade_off = self.analyze_trade_off()
        wacc_analysis = self.analyze_wacc()
        capacity = self.debt_capacity_analysis()

        # Use average of trade-off and WACC optimal leverage
        optimal = (trade_off['optimal_leverage'] + wacc_analysis['optimal_leverage']) / 2

        # Constrain by debt capacity
        max_leverage = capacity['debt_capacity'] / self.firm_value if self.firm_value > 0 else 0
        optimal = min(optimal, max_leverage)

        # Current values
        current_ts = tax_shield_value(self.debt, self.tax_rate)

        # Optimal values
        optimal_debt = self.firm_value * optimal
        optimal_ts = tax_shield_value(optimal_debt, self.tax_rate)

        # Distress cost
        prob_distress = distress_probability_from_leverage(optimal, self.asset_volatility)
        distress_cost = expected_distress_cost(prob_distress, 0.20, self.firm_value)

        # Net benefit
        net_benefit = optimal_ts - current_ts - distress_cost

        # Recommendation
        if optimal > self.current_leverage + 0.05:
            recommendation = f"Consider increasing debt. Target leverage: {optimal:.1%}"
        elif optimal < self.current_leverage - 0.05:
            recommendation = f"Consider reducing debt. Target leverage: {optimal:.1%}"
        else:
            recommendation = f"Current leverage is near optimal ({optimal:.1%})"

        return CapitalStructureResult(
            optimal_leverage=optimal,
            current_leverage=self.current_leverage,
            tax_shield_value=current_ts,
            expected_distress_cost=distress_cost,
            net_benefit=net_benefit,
            wacc_current=self.current_wacc,
            wacc_optimal=wacc_analysis['minimum_wacc'],
            recommendation=recommendation
        )

    def scenario_analysis(
        self,
        leverage_scenarios: List[float]
    ) -> List[Dict]:
        """
        Analyze multiple leverage scenarios.

        Args:
            leverage_scenarios: List of leverage ratios to analyze

        Returns:
            List of scenario results
        """
        results = []

        for leverage in leverage_scenarios:
            debt_amount = self.firm_value * leverage
            equity_amount = self.firm_value * (1 - leverage)

            # Costs increase with leverage
            cost_of_equity_adj = mm_proposition_2(
                self.cost_of_equity * 0.8,  # Approximate unlevered
                self.cost_of_debt,
                debt_amount,
                equity_amount,
                self.tax_rate
            )

            wacc = ((1 - leverage) * cost_of_equity_adj +
                    leverage * self.cost_of_debt * (1 - self.tax_rate))

            ts = tax_shield_value(debt_amount, self.tax_rate)
            prob_dist = distress_probability_from_leverage(leverage, self.asset_volatility)
            dist_cost = expected_distress_cost(prob_dist, 0.20, self.firm_value)

            results.append({
                'leverage': leverage,
                'debt': debt_amount,
                'equity': equity_amount,
                'cost_of_equity': cost_of_equity_adj,
                'wacc': wacc,
                'tax_shield': ts,
                'prob_distress': prob_dist,
                'distress_cost': dist_cost,
                'net_benefit': ts - dist_cost,
            })

        return results
