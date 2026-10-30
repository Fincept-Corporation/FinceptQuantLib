"""
Segment Analysis Module

Sum-of-the-Parts (SOTP) valuation and segment analysis:
- Business Segment Valuation
- Geographic Segment Analysis
- Conglomerate Discount/Premium
- Segment Profitability Analysis
- Cross-Segment Synergies
- Breakup Value Analysis
"""

from dataclasses import dataclass
from typing import Optional, Dict, List


# =============================================================================
# Segment Data Structures
# =============================================================================

@dataclass
class BusinessSegment:
    """Business segment financial data."""
    name: str
    revenue: float = 0.0
    operating_income: float = 0.0
    ebitda: float = 0.0
    assets: float = 0.0
    capital_employed: float = 0.0
    capex: float = 0.0
    depreciation: float = 0.0

    # Optional growth and margin data
    revenue_growth: Optional[float] = None
    operating_margin: Optional[float] = None

    # Comparable data for valuation
    comparable_ev_ebitda: Optional[float] = None
    comparable_ev_revenue: Optional[float] = None
    comparable_pe: Optional[float] = None

    def __post_init__(self):
        """Calculate derived metrics."""
        if self.operating_margin is None and self.revenue > 0:
            self.operating_margin = self.operating_income / self.revenue


@dataclass
class GeographicSegment:
    """Geographic segment financial data."""
    region: str
    revenue: float = 0.0
    operating_income: float = 0.0
    assets: float = 0.0
    employees: int = 0
    revenue_growth: Optional[float] = None


# =============================================================================
# Sum-of-the-Parts Valuation
# =============================================================================

@dataclass
class SOTPResult:
    """Sum-of-the-parts valuation result."""
    segment_values: Dict[str, float]
    gross_asset_value: float
    corporate_costs: float
    net_debt: float
    equity_value: float
    value_per_share: float
    implied_conglomerate_discount: Optional[float]


def segment_value_ev_ebitda(
    segment: BusinessSegment,
    multiple: Optional[float] = None
) -> float:
    """
    Value segment using EV/EBITDA multiple.

    Args:
        segment: Business segment data
        multiple: EV/EBITDA multiple (uses comparable if not provided)

    Returns:
        Segment enterprise value
    """
    mult = multiple or segment.comparable_ev_ebitda
    if mult is None or segment.ebitda <= 0:
        return 0.0
    return segment.ebitda * mult


def segment_value_ev_revenue(
    segment: BusinessSegment,
    multiple: Optional[float] = None
) -> float:
    """
    Value segment using EV/Revenue multiple.

    Useful for high-growth or unprofitable segments.

    Args:
        segment: Business segment data
        multiple: EV/Revenue multiple (uses comparable if not provided)

    Returns:
        Segment enterprise value
    """
    mult = multiple or segment.comparable_ev_revenue
    if mult is None or segment.revenue <= 0:
        return 0.0
    return segment.revenue * mult


def segment_value_dcf(
    segment: BusinessSegment,
    wacc: float,
    growth_rate: float,
    terminal_growth: float = 0.02,
    forecast_years: int = 5,
    tax_rate: float = 0.25
) -> float:
    """
    Value segment using DCF.

    Args:
        segment: Business segment data
        wacc: Weighted average cost of capital
        growth_rate: Revenue/earnings growth rate
        terminal_growth: Terminal growth rate
        forecast_years: Explicit forecast period
        tax_rate: Tax rate

    Returns:
        Segment enterprise value
    """
    # Starting NOPAT
    nopat = segment.operating_income * (1 - tax_rate)

    if nopat <= 0:
        return segment_value_ev_revenue(segment)

    # Forecast period
    pv_fcf = 0.0
    fcf = nopat
    for t in range(1, forecast_years + 1):
        fcf *= (1 + growth_rate)
        pv_fcf += fcf / ((1 + wacc) ** t)

    # Terminal value
    terminal_fcf = fcf * (1 + terminal_growth)
    if wacc > terminal_growth:
        terminal_value = terminal_fcf / (wacc - terminal_growth)
        pv_terminal = terminal_value / ((1 + wacc) ** forecast_years)
    else:
        pv_terminal = 0

    return pv_fcf + pv_terminal


def sum_of_the_parts(
    segments: List[BusinessSegment],
    corporate_overhead: float = 0.0,
    net_debt: float = 0.0,
    shares_outstanding: float = 1.0,
    valuation_method: str = 'ev_ebitda',
    custom_multiples: Optional[Dict[str, float]] = None
) -> SOTPResult:
    """
    Sum-of-the-Parts Valuation.

    Values each business segment separately and sums.

    Args:
        segments: List of business segments
        corporate_overhead: Annual corporate overhead costs
        net_debt: Net debt (debt - cash)
        shares_outstanding: Shares outstanding
        valuation_method: 'ev_ebitda', 'ev_revenue', or 'blended'
        custom_multiples: Custom multiples by segment name

    Returns:
        SOTPResult with full valuation
    """
    segment_values = {}

    for segment in segments:
        # Get custom multiple if provided
        custom_mult = None
        if custom_multiples and segment.name in custom_multiples:
            custom_mult = custom_multiples[segment.name]

        # Calculate segment value
        if valuation_method == 'ev_ebitda':
            value = segment_value_ev_ebitda(segment, custom_mult)
        elif valuation_method == 'ev_revenue':
            value = segment_value_ev_revenue(segment, custom_mult)
        elif valuation_method == 'blended':
            # 70% EBITDA, 30% Revenue
            ev_ebitda_val = segment_value_ev_ebitda(segment, custom_mult)
            ev_rev_val = segment_value_ev_revenue(segment, custom_mult)
            value = 0.7 * ev_ebitda_val + 0.3 * ev_rev_val
        else:
            value = segment_value_ev_ebitda(segment, custom_mult)

        segment_values[segment.name] = value

    # Gross asset value
    gross_value = sum(segment_values.values())

    # Capitalize corporate overhead (assume 10x multiple)
    corporate_cost_value = corporate_overhead * 10

    # Equity value
    equity_value = gross_value - corporate_cost_value - net_debt

    # Value per share
    value_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0

    return SOTPResult(
        segment_values=segment_values,
        gross_asset_value=gross_value,
        corporate_costs=corporate_cost_value,
        net_debt=net_debt,
        equity_value=equity_value,
        value_per_share=value_per_share,
        implied_conglomerate_discount=None  # Calculated separately if market cap known
    )


# =============================================================================
# Conglomerate Discount Analysis
# =============================================================================

def conglomerate_discount(
    market_cap: float,
    sotp_equity_value: float
) -> Dict[str, float]:
    """
    Calculate Conglomerate Discount/Premium.

    Discount = (SOTP Value - Market Cap) / SOTP Value

    Typical conglomerate discount: 10-20%

    Args:
        market_cap: Current market capitalization
        sotp_equity_value: Sum-of-parts equity value

    Returns:
        Dictionary with discount analysis
    """
    if sotp_equity_value <= 0:
        return {'discount': 0, 'premium': 0, 'ratio': 1}

    difference = sotp_equity_value - market_cap
    discount = difference / sotp_equity_value

    return {
        'sotp_value': sotp_equity_value,
        'market_cap': market_cap,
        'value_gap': difference,
        'discount': discount,  # Positive = trading below SOTP
        'market_to_sotp_ratio': market_cap / sotp_equity_value,
    }


def diversification_discount_model(
    segment_correlations: List[List[float]],
    segment_values: List[float],
    market_risk_aversion: float = 0.5
) -> float:
    """
    Theoretical Diversification Discount.

    Based on diversification reducing portfolio volatility,
    but potentially reducing value if market prefers focused firms.

    Args:
        segment_correlations: Correlation matrix between segments
        segment_values: Value of each segment
        market_risk_aversion: Market's preference for focus

    Returns:
        Theoretical discount percentage
    """
    if not segment_values or len(segment_values) < 2:
        return 0.0

    total_value = sum(segment_values)
    if total_value <= 0:
        return 0.0

    # Calculate diversification benefit (portfolio variance reduction)
    n = len(segment_values)
    weights = [v / total_value for v in segment_values]

    # Herfindahl index of business mix
    herfindahl = sum(w ** 2 for w in weights)

    # More diversified (lower Herfindahl) → larger discount
    # Max discount when perfectly diversified (Herfindahl = 1/n)
    # Zero discount when focused (Herfindahl = 1)
    max_diversification = 1 - (1 / n)
    current_diversification = 1 - herfindahl

    # Discount increases with diversification and risk aversion
    theoretical_discount = market_risk_aversion * (current_diversification / max_diversification) * 0.20

    return theoretical_discount


# =============================================================================
# Segment Profitability Analysis
# =============================================================================

def segment_profitability(
    segments: List[BusinessSegment]
) -> Dict[str, Dict[str, float]]:
    """
    Analyze Segment Profitability.

    Calculates key metrics for each segment.

    Args:
        segments: List of business segments

    Returns:
        Dictionary with profitability metrics by segment
    """
    results = {}

    total_revenue = sum(s.revenue for s in segments)
    total_operating_income = sum(s.operating_income for s in segments)

    for segment in segments:
        # Segment margins
        op_margin = segment.operating_income / segment.revenue if segment.revenue > 0 else 0
        ebitda_margin = segment.ebitda / segment.revenue if segment.revenue > 0 else 0

        # Asset efficiency
        asset_turnover = segment.revenue / segment.assets if segment.assets > 0 else 0
        roa = segment.operating_income / segment.assets if segment.assets > 0 else 0
        roic = segment.operating_income / segment.capital_employed if segment.capital_employed > 0 else 0

        # Contribution to total
        revenue_contribution = segment.revenue / total_revenue if total_revenue > 0 else 0
        profit_contribution = segment.operating_income / total_operating_income if total_operating_income > 0 else 0

        # Relative profitability (profit contribution / revenue contribution)
        relative_profitability = profit_contribution / revenue_contribution if revenue_contribution > 0 else 0

        results[segment.name] = {
            'revenue': segment.revenue,
            'operating_income': segment.operating_income,
            'operating_margin': op_margin,
            'ebitda_margin': ebitda_margin,
            'asset_turnover': asset_turnover,
            'roa': roa,
            'roic': roic,
            'revenue_contribution': revenue_contribution,
            'profit_contribution': profit_contribution,
            'relative_profitability': relative_profitability,
        }

    return results


def segment_growth_analysis(
    segments_current: List[BusinessSegment],
    segments_prior: List[BusinessSegment]
) -> Dict[str, Dict[str, float]]:
    """
    Analyze Segment Growth Rates.

    Compares current and prior period by segment.

    Args:
        segments_current: Current period segments
        segments_prior: Prior period segments

    Returns:
        Dictionary with growth metrics by segment
    """
    # Create lookup for prior period
    prior_lookup = {s.name: s for s in segments_prior}

    results = {}

    for segment in segments_current:
        prior = prior_lookup.get(segment.name)

        if prior:
            rev_growth = (segment.revenue - prior.revenue) / prior.revenue if prior.revenue > 0 else 0
            op_income_growth = (segment.operating_income - prior.operating_income) / prior.operating_income if prior.operating_income > 0 else 0
            margin_change = (segment.operating_margin or 0) - (prior.operating_margin or 0)
        else:
            rev_growth = None
            op_income_growth = None
            margin_change = None

        results[segment.name] = {
            'revenue_growth': rev_growth,
            'operating_income_growth': op_income_growth,
            'margin_change': margin_change,
            'current_revenue': segment.revenue,
            'current_operating_income': segment.operating_income,
        }

    return results


# =============================================================================
# Breakup Analysis
# =============================================================================

@dataclass
class BreakupAnalysis:
    """Breakup value analysis result."""
    sotp_value: float
    separation_costs: float
    dis_synergies: float
    tax_leakage: float
    net_breakup_value: float
    value_creation: float
    recommendation: str


def breakup_value_analysis(
    segments: List[BusinessSegment],
    market_cap: float,
    net_debt: float,
    shares_outstanding: float,
    separation_costs_pct: float = 0.03,
    dis_synergy_pct: float = 0.05,
    tax_leakage_pct: float = 0.02
) -> BreakupAnalysis:
    """
    Analyze Potential Breakup Value.

    Estimates value that could be unlocked by separating business units.

    Args:
        segments: List of business segments
        market_cap: Current market cap
        net_debt: Net debt
        shares_outstanding: Shares outstanding
        separation_costs_pct: Transaction costs as % of value
        dis_synergy_pct: Lost synergies as % of value
        tax_leakage_pct: Tax costs as % of value

    Returns:
        BreakupAnalysis with value creation estimate
    """
    # SOTP valuation
    sotp = sum_of_the_parts(
        segments,
        corporate_overhead=0,  # Assume corporate costs eliminated
        net_debt=net_debt,
        shares_outstanding=shares_outstanding
    )

    gross_sotp = sotp.gross_asset_value

    # Costs of separation
    separation_costs = gross_sotp * separation_costs_pct
    dis_synergies = gross_sotp * dis_synergy_pct
    tax_leakage = gross_sotp * tax_leakage_pct

    # Net breakup value
    net_breakup = gross_sotp - separation_costs - dis_synergies - tax_leakage - net_debt

    # Value creation vs current
    value_creation = net_breakup - market_cap
    value_creation_pct = value_creation / market_cap if market_cap > 0 else 0

    # Recommendation
    if value_creation_pct > 0.15:
        recommendation = "Strong breakup candidate - significant value unlock potential"
    elif value_creation_pct > 0.05:
        recommendation = "Moderate breakup potential - consider strategic alternatives"
    elif value_creation_pct > 0:
        recommendation = "Marginal breakup value - benefits may not exceed costs"
    else:
        recommendation = "No breakup value - conglomerate structure adds value"

    return BreakupAnalysis(
        sotp_value=gross_sotp,
        separation_costs=separation_costs,
        dis_synergies=dis_synergies,
        tax_leakage=tax_leakage,
        net_breakup_value=net_breakup,
        value_creation=value_creation,
        recommendation=recommendation,
    )


# =============================================================================
# Geographic Segment Analysis
# =============================================================================

def geographic_segment_analysis(
    segments: List[GeographicSegment]
) -> Dict[str, Dict[str, float]]:
    """
    Analyze Geographic Segments.

    Args:
        segments: List of geographic segments

    Returns:
        Dictionary with analysis by region
    """
    total_revenue = sum(s.revenue for s in segments)
    total_assets = sum(s.assets for s in segments)

    results = {}

    for segment in segments:
        rev_share = segment.revenue / total_revenue if total_revenue > 0 else 0
        asset_share = segment.assets / total_assets if total_assets > 0 else 0

        margin = segment.operating_income / segment.revenue if segment.revenue > 0 else 0
        asset_efficiency = segment.revenue / segment.assets if segment.assets > 0 else 0

        results[segment.region] = {
            'revenue': segment.revenue,
            'revenue_share': rev_share,
            'operating_income': segment.operating_income,
            'operating_margin': margin,
            'assets': segment.assets,
            'asset_share': asset_share,
            'revenue_per_employee': segment.revenue / segment.employees if segment.employees > 0 else 0,
            'asset_efficiency': asset_efficiency,
            'growth': segment.revenue_growth,
        }

    return results


def geographic_risk_assessment(
    segments: List[GeographicSegment],
    region_risk_scores: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Assess Geographic Concentration Risk.

    Args:
        segments: Geographic segments
        region_risk_scores: Risk scores by region (0-1, higher = more risk)

    Returns:
        Risk assessment metrics
    """
    # Default risk scores
    if region_risk_scores is None:
        region_risk_scores = {
            'North America': 0.2,
            'Western Europe': 0.25,
            'Japan': 0.25,
            'Australia': 0.2,
            'China': 0.5,
            'Emerging Asia': 0.5,
            'Latin America': 0.55,
            'Eastern Europe': 0.5,
            'Middle East': 0.6,
            'Africa': 0.65,
        }

    total_revenue = sum(s.revenue for s in segments)
    if total_revenue <= 0:
        return {'herfindahl': 0, 'weighted_risk': 0}

    # Calculate metrics
    revenue_weights = [s.revenue / total_revenue for s in segments]
    herfindahl = sum(w ** 2 for w in revenue_weights)

    # Weighted average risk
    weighted_risk = 0.0
    for segment in segments:
        weight = segment.revenue / total_revenue
        risk = region_risk_scores.get(segment.region, 0.4)
        weighted_risk += weight * risk

    # Concentration risk (higher Herfindahl = more concentrated)
    concentration_risk = 'high' if herfindahl > 0.5 else ('medium' if herfindahl > 0.25 else 'low')

    return {
        'herfindahl_index': herfindahl,
        'weighted_risk_score': weighted_risk,
        'concentration_risk': concentration_risk,
        'number_of_regions': len(segments),
        'largest_region_share': max(revenue_weights) if revenue_weights else 0,
    }


# =============================================================================
# Segment Analysis Class
# =============================================================================

class SegmentAnalysis:
    """
    Comprehensive Segment Analysis.

    Provides SOTP valuation, breakup analysis, and segment profitability.
    """

    def __init__(
        self,
        business_segments: List[BusinessSegment],
        geographic_segments: Optional[List[GeographicSegment]] = None,
        market_cap: float = 0.0,
        net_debt: float = 0.0,
        shares_outstanding: float = 1.0,
        corporate_overhead: float = 0.0
    ):
        """Initialize segment analysis."""
        self.business_segments = business_segments
        self.geographic_segments = geographic_segments or []
        self.market_cap = market_cap
        self.net_debt = net_debt
        self.shares = shares_outstanding
        self.corporate_overhead = corporate_overhead

    def sotp_valuation(
        self,
        method: str = 'ev_ebitda',
        custom_multiples: Optional[Dict[str, float]] = None
    ) -> SOTPResult:
        """Perform sum-of-the-parts valuation."""
        result = sum_of_the_parts(
            self.business_segments,
            self.corporate_overhead,
            self.net_debt,
            self.shares,
            method,
            custom_multiples
        )

        # Calculate implied discount if market cap available
        if self.market_cap > 0 and result.equity_value > 0:
            discount_info = conglomerate_discount(self.market_cap, result.equity_value)
            result.implied_conglomerate_discount = discount_info['discount']

        return result

    def profitability_analysis(self) -> Dict[str, Dict[str, float]]:
        """Analyze segment profitability."""
        return segment_profitability(self.business_segments)

    def breakup_analysis(
        self,
        separation_costs: float = 0.03,
        dis_synergies: float = 0.05
    ) -> BreakupAnalysis:
        """Analyze potential breakup value."""
        return breakup_value_analysis(
            self.business_segments,
            self.market_cap,
            self.net_debt,
            self.shares,
            separation_costs,
            dis_synergies
        )

    def geographic_analysis(self) -> Dict:
        """Analyze geographic segments."""
        if not self.geographic_segments:
            return {}

        return {
            'segment_metrics': geographic_segment_analysis(self.geographic_segments),
            'risk_assessment': geographic_risk_assessment(self.geographic_segments),
        }

    def full_analysis(self) -> Dict:
        """
        Perform comprehensive segment analysis.

        Returns:
            Dictionary with all analysis components
        """
        return {
            'sotp_valuation': self.sotp_valuation(),
            'profitability': self.profitability_analysis(),
            'breakup_analysis': self.breakup_analysis(),
            'geographic_analysis': self.geographic_analysis() if self.geographic_segments else None,
        }

    def segment_contribution_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Create segment contribution matrix.

        Shows each segment's contribution to revenue, profit, assets.

        Returns:
            Matrix of contributions
        """
        total_revenue = sum(s.revenue for s in self.business_segments)
        total_profit = sum(s.operating_income for s in self.business_segments)
        total_assets = sum(s.assets for s in self.business_segments)
        total_capital = sum(s.capital_employed for s in self.business_segments)

        matrix = {}
        for segment in self.business_segments:
            matrix[segment.name] = {
                'revenue_contribution': segment.revenue / total_revenue if total_revenue > 0 else 0,
                'profit_contribution': segment.operating_income / total_profit if total_profit > 0 else 0,
                'asset_contribution': segment.assets / total_assets if total_assets > 0 else 0,
                'capital_contribution': segment.capital_employed / total_capital if total_capital > 0 else 0,
            }

        return matrix
