"""
Comparable Analysis (Peer Comparison)

Comprehensive peer analysis:
- Peer Group Comparison
- Relative Valuation
- Percentile Ranking
- Z-Score Normalization
- Screening & Filtering
- Sector Benchmarking

These tools help analyze a company relative to its peers.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
from .statements import (
    IncomeStatement, BalanceSheet, FinancialStatements
)
from .valuation import MarketData


@dataclass
class PeerMetrics:
    """Metrics for a single company in peer comparison."""
    name: str = ""
    ticker: str = ""

    # Valuation
    pe_ratio: float = 0.0
    pb_ratio: float = 0.0
    ps_ratio: float = 0.0
    ev_ebitda: float = 0.0
    ev_revenue: float = 0.0

    # Profitability
    gross_margin: float = 0.0
    operating_margin: float = 0.0
    net_margin: float = 0.0
    roe: float = 0.0
    roic: float = 0.0

    # Growth
    revenue_growth: float = 0.0
    earnings_growth: float = 0.0

    # Financial Health
    debt_to_equity: float = 0.0
    current_ratio: float = 0.0


@dataclass
class ComparableResult:
    """Result of comparable analysis."""
    target: PeerMetrics = None
    peers: List[PeerMetrics] = field(default_factory=list)
    peer_median: Dict[str, float] = field(default_factory=dict)
    peer_mean: Dict[str, float] = field(default_factory=dict)
    target_percentile: Dict[str, float] = field(default_factory=dict)
    implied_values: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScreeningCriteria:
    """Criteria for stock screening."""
    pe_max: Optional[float] = None
    pe_min: Optional[float] = None
    pb_max: Optional[float] = None
    pb_min: Optional[float] = None
    roe_min: Optional[float] = None
    debt_equity_max: Optional[float] = None
    current_ratio_min: Optional[float] = None
    gross_margin_min: Optional[float] = None
    revenue_growth_min: Optional[float] = None
    market_cap_min: Optional[float] = None
    market_cap_max: Optional[float] = None


# =============================================================================
# PEER COMPARISON UTILITIES
# =============================================================================

def calculate_peer_metrics(
    statements: FinancialStatements,
    market: MarketData,
    prior: Optional[FinancialStatements] = None,
    name: str = "",
    ticker: str = ""
) -> PeerMetrics:
    """
    Calculate all standard metrics for a single company.

    Used to create comparable peer metrics.
    """
    income = statements.income_statement
    balance = statements.balance_sheet
    prior_income = prior.income_statement if prior else None
    prior_balance = prior.balance_sheet if prior else None

    # Valuation metrics
    pe = market.stock_price / income.eps_basic if income.eps_basic > 0 else float('inf')
    pb = market.stock_price / (balance.total_equity / market.shares_outstanding) if balance.total_equity > 0 else float('inf')
    ps = market.market_cap / income.revenue if income.revenue > 0 else float('inf')

    ev = market.market_cap + balance.total_debt - balance.cash_and_equivalents
    ev_ebitda = ev / income.ebitda if income.ebitda > 0 else float('inf')
    ev_rev = ev / income.revenue if income.revenue > 0 else float('inf')

    # Profitability
    gross_margin = income.gross_profit / income.revenue if income.revenue > 0 else 0
    op_margin = income.operating_income / income.revenue if income.revenue > 0 else 0
    net_margin = income.net_income / income.revenue if income.revenue > 0 else 0

    avg_equity = balance.total_equity
    if prior_balance:
        avg_equity = (balance.total_equity + prior_balance.total_equity) / 2
    roe = income.net_income / avg_equity if avg_equity > 0 else 0

    avg_ic = balance.invested_capital
    if prior_balance:
        avg_ic = (balance.invested_capital + prior_balance.invested_capital) / 2
    tax_rate = income.effective_tax_rate
    nopat = income.operating_income * (1 - tax_rate)
    roic = nopat / avg_ic if avg_ic > 0 else 0

    # Growth
    rev_growth = 0.0
    earn_growth = 0.0
    if prior_income:
        if prior_income.revenue > 0:
            rev_growth = (income.revenue - prior_income.revenue) / prior_income.revenue
        if prior_income.net_income > 0:
            earn_growth = (income.net_income - prior_income.net_income) / prior_income.net_income

    # Financial health
    de = balance.total_debt / balance.total_equity if balance.total_equity > 0 else float('inf')
    cr = balance.total_current_assets / balance.total_current_liabilities if balance.total_current_liabilities > 0 else float('inf')

    return PeerMetrics(
        name=name,
        ticker=ticker,
        pe_ratio=pe,
        pb_ratio=pb,
        ps_ratio=ps,
        ev_ebitda=ev_ebitda,
        ev_revenue=ev_rev,
        gross_margin=gross_margin,
        operating_margin=op_margin,
        net_margin=net_margin,
        roe=roe,
        roic=roic,
        revenue_growth=rev_growth,
        earnings_growth=earn_growth,
        debt_to_equity=de,
        current_ratio=cr,
    )


def calculate_percentile(value: float, values: List[float]) -> float:
    """
    Calculate percentile rank of a value within a distribution.

    Returns percentile as 0-100.
    """
    if not values:
        return 50.0

    count_below = sum(1 for v in values if v < value)
    count_equal = sum(1 for v in values if v == value)

    # Percentile using linear interpolation
    percentile = (count_below + 0.5 * count_equal) / len(values) * 100

    return percentile


def calculate_z_score(value: float, values: List[float]) -> float:
    """
    Calculate z-score (standard deviations from mean).

    Returns z-score.
    """
    if not values:
        return 0.0

    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance ** 0.5

    if std == 0:
        return 0.0

    return (value - mean) / std


# =============================================================================
# PEER COMPARISON ANALYSIS
# =============================================================================

def peer_comparison(
    target: PeerMetrics,
    peers: List[PeerMetrics],
    metrics: Optional[List[str]] = None
) -> ComparableResult:
    """
    Compare target company against peer group.

    Args:
        target: Target company metrics
        peers: List of peer company metrics
        metrics: List of metric names to compare (defaults to all)

    Returns: ComparableResult with comparison data
    """
    if metrics is None:
        metrics = [
            'pe_ratio', 'pb_ratio', 'ps_ratio', 'ev_ebitda', 'ev_revenue',
            'gross_margin', 'operating_margin', 'net_margin', 'roe', 'roic',
            'revenue_growth', 'earnings_growth', 'debt_to_equity', 'current_ratio'
        ]

    peer_median = {}
    peer_mean = {}
    target_percentile = {}

    for metric in metrics:
        target_val = getattr(target, metric, 0.0)
        peer_values = [getattr(p, metric, 0.0) for p in peers]

        # Filter out infinities and extreme values
        valid_values = [v for v in peer_values if -1000 < v < 1000 and v != float('inf') and v != float('-inf')]

        if valid_values:
            sorted_vals = sorted(valid_values)
            n = len(sorted_vals)
            median = sorted_vals[n // 2] if n % 2 == 1 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
            mean = sum(valid_values) / n

            peer_median[metric] = median
            peer_mean[metric] = mean
            target_percentile[metric] = calculate_percentile(target_val, valid_values)
        else:
            peer_median[metric] = 0.0
            peer_mean[metric] = 0.0
            target_percentile[metric] = 50.0

    return ComparableResult(
        target=target,
        peers=peers,
        peer_median=peer_median,
        peer_mean=peer_mean,
        target_percentile=target_percentile,
    )


def relative_valuation(
    target: PeerMetrics,
    peers: List[PeerMetrics],
    target_income: IncomeStatement,
    target_balance: BalanceSheet,
    target_market: MarketData
) -> Dict[str, float]:
    """
    Calculate implied values based on peer multiples.

    Returns implied share prices based on different multiples.
    """
    # Get peer medians
    comparison = peer_comparison(target, peers)

    implied_values = {}

    # P/E implied value
    if comparison.peer_median.get('pe_ratio', 0) > 0 and target_income.eps_basic > 0:
        implied_values['pe_implied'] = comparison.peer_median['pe_ratio'] * target_income.eps_basic

    # P/B implied value
    if comparison.peer_median.get('pb_ratio', 0) > 0 and target_balance.total_equity > 0:
        bvps = target_balance.total_equity / target_market.shares_outstanding
        implied_values['pb_implied'] = comparison.peer_median['pb_ratio'] * bvps

    # P/S implied value
    if comparison.peer_median.get('ps_ratio', 0) > 0 and target_income.revenue > 0:
        rps = target_income.revenue / target_market.shares_outstanding
        implied_values['ps_implied'] = comparison.peer_median['ps_ratio'] * rps

    # EV/EBITDA implied value
    if comparison.peer_median.get('ev_ebitda', 0) > 0 and target_income.ebitda > 0:
        implied_ev = comparison.peer_median['ev_ebitda'] * target_income.ebitda
        implied_equity = implied_ev - target_balance.total_debt + target_balance.cash_and_equivalents
        implied_values['ev_ebitda_implied'] = implied_equity / target_market.shares_outstanding

    # Average implied value
    valid_implied = [v for v in implied_values.values() if v > 0]
    if valid_implied:
        implied_values['average_implied'] = sum(valid_implied) / len(valid_implied)

    # Premium/discount to current
    if 'average_implied' in implied_values and target_market.stock_price > 0:
        implied_values['premium_discount'] = (implied_values['average_implied'] - target_market.stock_price) / target_market.stock_price

    return implied_values


def percentile_rank(
    target: PeerMetrics,
    peers: List[PeerMetrics]
) -> Dict[str, float]:
    """
    Calculate percentile rank for all metrics.

    Returns dictionary of metric -> percentile (0-100).
    """
    comparison = peer_comparison(target, peers)
    return comparison.target_percentile


# =============================================================================
# SCREENING
# =============================================================================

def screen_companies(
    companies: List[PeerMetrics],
    criteria: ScreeningCriteria
) -> List[PeerMetrics]:
    """
    Screen companies based on criteria.

    Args:
        companies: List of companies to screen
        criteria: Screening criteria

    Returns: List of companies passing all criteria
    """
    results = []

    for company in companies:
        passes = True

        # P/E criteria
        if criteria.pe_max is not None and company.pe_ratio > criteria.pe_max:
            passes = False
        if criteria.pe_min is not None and company.pe_ratio < criteria.pe_min:
            passes = False

        # P/B criteria
        if criteria.pb_max is not None and company.pb_ratio > criteria.pb_max:
            passes = False
        if criteria.pb_min is not None and company.pb_ratio < criteria.pb_min:
            passes = False

        # ROE criteria
        if criteria.roe_min is not None and company.roe < criteria.roe_min:
            passes = False

        # Debt/Equity criteria
        if criteria.debt_equity_max is not None and company.debt_to_equity > criteria.debt_equity_max:
            passes = False

        # Current ratio criteria
        if criteria.current_ratio_min is not None and company.current_ratio < criteria.current_ratio_min:
            passes = False

        # Gross margin criteria
        if criteria.gross_margin_min is not None and company.gross_margin < criteria.gross_margin_min:
            passes = False

        # Revenue growth criteria
        if criteria.revenue_growth_min is not None and company.revenue_growth < criteria.revenue_growth_min:
            passes = False

        if passes:
            results.append(company)

    return results


def value_screen(companies: List[PeerMetrics]) -> List[PeerMetrics]:
    """Pre-built value investing screen."""
    criteria = ScreeningCriteria(
        pe_max=15.0,
        pb_max=2.0,
        roe_min=0.10,
        debt_equity_max=1.0,
        current_ratio_min=1.5,
    )
    return screen_companies(companies, criteria)


def growth_screen(companies: List[PeerMetrics]) -> List[PeerMetrics]:
    """Pre-built growth investing screen."""
    criteria = ScreeningCriteria(
        revenue_growth_min=0.15,
        gross_margin_min=0.30,
        roe_min=0.15,
    )
    return screen_companies(companies, criteria)


def quality_screen(companies: List[PeerMetrics]) -> List[PeerMetrics]:
    """Pre-built quality investing screen."""
    criteria = ScreeningCriteria(
        roe_min=0.15,
        gross_margin_min=0.40,
        debt_equity_max=0.5,
        current_ratio_min=2.0,
    )
    return screen_companies(companies, criteria)


# =============================================================================
# COMPREHENSIVE ANALYSIS CLASS
# =============================================================================

class ComparableAnalysis:
    """
    Comprehensive comparable/peer analysis.

    Example:
        # Create target and peer metrics
        target = calculate_peer_metrics(target_statements, target_market, name="Apple")
        peers = [calculate_peer_metrics(p_stmt, p_mkt, name=p_name) for ...]

        # Run analysis
        analysis = ComparableAnalysis(target, peers)
        result = analysis.compare()

        # Get implied valuation
        implied = analysis.relative_value(target_income, target_balance, target_market)
        print(f"Average Implied Price: ${implied['average_implied']:.2f}")

        # Screen for value
        value_names = analysis.apply_screen(ScreeningCriteria(pe_max=15, pb_max=2))
    """

    def __init__(
        self,
        target: PeerMetrics,
        peers: List[PeerMetrics]
    ):
        self.target = target
        self.peers = peers

    def compare(self, metrics: Optional[List[str]] = None) -> ComparableResult:
        """Run full peer comparison."""
        return peer_comparison(self.target, self.peers, metrics)

    def relative_value(
        self,
        target_income: IncomeStatement,
        target_balance: BalanceSheet,
        target_market: MarketData
    ) -> Dict[str, float]:
        """Calculate relative valuation."""
        return relative_valuation(
            self.target, self.peers,
            target_income, target_balance, target_market
        )

    def percentiles(self) -> Dict[str, float]:
        """Get target's percentile rankings."""
        return percentile_rank(self.target, self.peers)

    def apply_screen(self, criteria: ScreeningCriteria) -> List[str]:
        """Apply screening criteria, return names of passing companies."""
        all_companies = [self.target] + self.peers
        passing = screen_companies(all_companies, criteria)
        return [c.name for c in passing]

    def z_scores(self) -> Dict[str, float]:
        """Calculate z-scores for all metrics."""
        metrics = [
            'pe_ratio', 'pb_ratio', 'ps_ratio', 'ev_ebitda', 'ev_revenue',
            'gross_margin', 'operating_margin', 'net_margin', 'roe', 'roic'
        ]

        z_scores = {}
        for metric in metrics:
            target_val = getattr(self.target, metric, 0.0)
            peer_values = [getattr(p, metric, 0.0) for p in self.peers]
            valid_values = [v for v in peer_values if -1000 < v < 1000 and v != float('inf')]

            if valid_values:
                z_scores[metric] = calculate_z_score(target_val, valid_values)
            else:
                z_scores[metric] = 0.0

        return z_scores

    def summary(self) -> str:
        """Generate text summary of peer comparison."""
        result = self.compare()
        percentiles = self.percentiles()

        summary = f"PEER COMPARISON: {self.target.name}\n"
        summary += "=" * 50 + "\n\n"
        summary += f"Compared against {len(self.peers)} peers\n\n"

        summary += "VALUATION VS PEERS:\n"
        summary += "-" * 30 + "\n"
        summary += f"  P/E Ratio:    {self.target.pe_ratio:>8.1f}x  (Median: {result.peer_median.get('pe_ratio', 0):.1f}x) [{percentiles.get('pe_ratio', 50):.0f}th pctl]\n"
        summary += f"  P/B Ratio:    {self.target.pb_ratio:>8.1f}x  (Median: {result.peer_median.get('pb_ratio', 0):.1f}x) [{percentiles.get('pb_ratio', 50):.0f}th pctl]\n"
        summary += f"  EV/EBITDA:    {self.target.ev_ebitda:>8.1f}x  (Median: {result.peer_median.get('ev_ebitda', 0):.1f}x) [{percentiles.get('ev_ebitda', 50):.0f}th pctl]\n\n"

        summary += "PROFITABILITY VS PEERS:\n"
        summary += "-" * 30 + "\n"
        summary += f"  Gross Margin:    {self.target.gross_margin:>7.1%}  (Median: {result.peer_median.get('gross_margin', 0):.1%})\n"
        summary += f"  Operating Margin:{self.target.operating_margin:>7.1%}  (Median: {result.peer_median.get('operating_margin', 0):.1%})\n"
        summary += f"  ROE:             {self.target.roe:>7.1%}  (Median: {result.peer_median.get('roe', 0):.1%})\n\n"

        # Relative positioning
        summary += "RELATIVE POSITIONING:\n"
        summary += "-" * 30 + "\n"

        # Determine if cheap or expensive
        val_pctls = [percentiles.get('pe_ratio', 50), percentiles.get('pb_ratio', 50), percentiles.get('ev_ebitda', 50)]
        avg_val_pctl = sum(val_pctls) / len(val_pctls)

        if avg_val_pctl < 30:
            summary += "  Valuation: CHEAP relative to peers\n"
        elif avg_val_pctl < 70:
            summary += "  Valuation: IN-LINE with peers\n"
        else:
            summary += "  Valuation: EXPENSIVE relative to peers\n"

        # Quality positioning
        quality_pctls = [percentiles.get('roe', 50), percentiles.get('gross_margin', 50)]
        avg_quality_pctl = sum(quality_pctls) / len(quality_pctls)

        if avg_quality_pctl > 70:
            summary += "  Quality: ABOVE-AVERAGE profitability\n"
        elif avg_quality_pctl > 30:
            summary += "  Quality: AVERAGE profitability\n"
        else:
            summary += "  Quality: BELOW-AVERAGE profitability\n"

        return summary
