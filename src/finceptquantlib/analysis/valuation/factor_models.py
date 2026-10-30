"""
Factor Models from Fundamentals

Factor construction and analysis using accounting data:
- Fama-French Style Factors (Value, Size, Profitability, Investment)
- Quality Factors
- Momentum Proxies from Fundamentals
- Composite Factor Scores
- Factor Exposure Analysis
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import math


# =============================================================================
# Company Fundamentals Data
# =============================================================================

@dataclass
class CompanyFundamentals:
    """Company fundamental data for factor analysis."""
    ticker: str

    # Size
    market_cap: float = 0.0
    total_assets: float = 0.0
    revenue: float = 0.0

    # Value Metrics
    book_value: float = 0.0
    earnings: float = 0.0
    cash_flow: float = 0.0
    dividends: float = 0.0
    sales: float = 0.0

    # Profitability
    gross_profit: float = 0.0
    operating_income: float = 0.0
    net_income: float = 0.0
    roe: float = 0.0
    roa: float = 0.0

    # Investment/Growth
    asset_growth: float = 0.0
    capex: float = 0.0
    rd_expense: float = 0.0

    # Quality
    accruals: float = 0.0
    earnings_variability: float = 0.0
    debt_to_equity: float = 0.0

    # Momentum (price-based, typically from returns)
    return_12m: Optional[float] = None
    return_6m: Optional[float] = None

    # Other
    beta: float = 1.0
    shares_outstanding: float = 0.0
    stock_price: float = 0.0

    @property
    def book_to_market(self) -> Optional[float]:
        """Book-to-market ratio."""
        if self.market_cap > 0:
            return self.book_value / self.market_cap
        return None

    @property
    def earnings_yield(self) -> Optional[float]:
        """Earnings yield (E/P)."""
        if self.market_cap > 0:
            return self.earnings / self.market_cap
        return None

    @property
    def gross_profitability(self) -> Optional[float]:
        """Gross profit / Assets."""
        if self.total_assets > 0:
            return self.gross_profit / self.total_assets
        return None


# =============================================================================
# Value Factors
# =============================================================================

def book_to_market_factor(
    book_value: float,
    market_cap: float
) -> Optional[float]:
    """
    Book-to-Market Ratio (HML Factor Component).

    B/M = Book Value of Equity / Market Value of Equity

    High B/M = Value stock
    Low B/M = Growth stock

    Args:
        book_value: Book value of equity
        market_cap: Market capitalization

    Returns:
        Book-to-market ratio
    """
    if market_cap <= 0:
        return None
    return book_value / market_cap


def earnings_yield_factor(
    earnings: float,
    market_cap: float
) -> Optional[float]:
    """
    Earnings Yield (E/P).

    Alternative value factor. Inverse of P/E.

    Args:
        earnings: Net income or operating earnings
        market_cap: Market capitalization

    Returns:
        Earnings yield
    """
    if market_cap <= 0:
        return None
    return earnings / market_cap


def cash_flow_yield_factor(
    operating_cash_flow: float,
    market_cap: float
) -> Optional[float]:
    """
    Cash Flow Yield.

    CFO / Market Cap

    Args:
        operating_cash_flow: Cash from operations
        market_cap: Market capitalization

    Returns:
        Cash flow yield
    """
    if market_cap <= 0:
        return None
    return operating_cash_flow / market_cap


def dividend_yield_factor(
    dividends: float,
    market_cap: float
) -> Optional[float]:
    """
    Dividend Yield.

    Args:
        dividends: Total dividends paid
        market_cap: Market capitalization

    Returns:
        Dividend yield
    """
    if market_cap <= 0:
        return None
    return dividends / market_cap


def sales_to_price_factor(
    sales: float,
    market_cap: float
) -> Optional[float]:
    """
    Sales-to-Price Ratio.

    S/P = Revenue / Market Cap (inverse of P/S)

    Args:
        sales: Total revenue
        market_cap: Market capitalization

    Returns:
        Sales-to-price ratio
    """
    if market_cap <= 0:
        return None
    return sales / market_cap


def composite_value_score(
    book_to_market: Optional[float],
    earnings_yield: Optional[float],
    cash_flow_yield: Optional[float],
    sales_to_price: Optional[float],
    weights: Optional[Dict[str, float]] = None
) -> Optional[float]:
    """
    Composite Value Score.

    Combines multiple value metrics into single score.

    Args:
        book_to_market: B/M ratio
        earnings_yield: E/P ratio
        cash_flow_yield: CF/P ratio
        sales_to_price: S/P ratio
        weights: Optional custom weights

    Returns:
        Composite value score
    """
    if weights is None:
        weights = {
            'book_to_market': 0.30,
            'earnings_yield': 0.30,
            'cash_flow_yield': 0.25,
            'sales_to_price': 0.15,
        }

    factors = {
        'book_to_market': book_to_market,
        'earnings_yield': earnings_yield,
        'cash_flow_yield': cash_flow_yield,
        'sales_to_price': sales_to_price,
    }

    # Only use available factors
    total_weight = 0.0
    weighted_sum = 0.0

    for name, value in factors.items():
        if value is not None:
            w = weights.get(name, 0.25)
            weighted_sum += w * value
            total_weight += w

    if total_weight <= 0:
        return None

    return weighted_sum / total_weight


# =============================================================================
# Size Factor
# =============================================================================

def size_factor(
    market_cap: float,
    log_transform: bool = True
) -> float:
    """
    Size Factor (SMB Component).

    Smaller firms historically outperform larger firms.

    Args:
        market_cap: Market capitalization
        log_transform: Use log of market cap

    Returns:
        Size factor (negative of log market cap for SMB)
    """
    if market_cap <= 0:
        return 0.0

    if log_transform:
        # Negative so smaller = higher factor value
        return -math.log(market_cap)
    else:
        return -market_cap


def total_assets_size(
    total_assets: float,
    log_transform: bool = True
) -> float:
    """
    Alternative Size Factor using Total Assets.

    Args:
        total_assets: Total assets
        log_transform: Use log transform

    Returns:
        Size factor based on assets
    """
    if total_assets <= 0:
        return 0.0

    if log_transform:
        return -math.log(total_assets)
    else:
        return -total_assets


# =============================================================================
# Profitability Factors
# =============================================================================

def gross_profitability_factor(
    gross_profit: float,
    total_assets: float
) -> Optional[float]:
    """
    Gross Profitability (RMW Component).

    Novy-Marx (2013): GP/A is strongest profitability predictor.

    Args:
        gross_profit: Gross profit
        total_assets: Total assets

    Returns:
        Gross profitability
    """
    if total_assets <= 0:
        return None
    return gross_profit / total_assets


def operating_profitability_factor(
    operating_income: float,
    book_equity: float
) -> Optional[float]:
    """
    Operating Profitability.

    Operating Income / Book Equity

    Args:
        operating_income: EBIT
        book_equity: Book value of equity

    Returns:
        Operating profitability
    """
    if book_equity <= 0:
        return None
    return operating_income / book_equity


def roe_factor(
    net_income: float,
    book_equity: float
) -> Optional[float]:
    """
    Return on Equity Factor.

    Args:
        net_income: Net income
        book_equity: Book value of equity

    Returns:
        ROE
    """
    if book_equity <= 0:
        return None
    return net_income / book_equity


def roa_factor(
    net_income: float,
    total_assets: float
) -> Optional[float]:
    """
    Return on Assets Factor.

    Args:
        net_income: Net income
        total_assets: Total assets

    Returns:
        ROA
    """
    if total_assets <= 0:
        return None
    return net_income / total_assets


def cash_based_profitability(
    cash_from_operations: float,
    total_assets: float
) -> Optional[float]:
    """
    Cash-Based Profitability.

    CFO / Total Assets

    More reliable than accrual-based measures.

    Args:
        cash_from_operations: Operating cash flow
        total_assets: Total assets

    Returns:
        Cash profitability
    """
    if total_assets <= 0:
        return None
    return cash_from_operations / total_assets


# =============================================================================
# Investment Factor
# =============================================================================

def asset_growth_factor(
    current_assets: float,
    prior_assets: float
) -> Optional[float]:
    """
    Asset Growth Factor (CMA Component).

    Conservative firms (low asset growth) outperform aggressive firms.

    ΔAssets = (Assets_t - Assets_t-1) / Assets_t-1

    Lower (more negative) asset growth = higher expected returns

    Args:
        current_assets: Current period total assets
        prior_assets: Prior period total assets

    Returns:
        Asset growth rate (negative for factor scoring)
    """
    if prior_assets <= 0:
        return None
    growth = (current_assets - prior_assets) / prior_assets
    return -growth  # Negative so conservative = higher score


def investment_rate_factor(
    capex: float,
    total_assets: float
) -> Optional[float]:
    """
    Investment Rate Factor.

    CapEx / Total Assets

    Lower investment = higher expected returns (CMA style)

    Args:
        capex: Capital expenditures
        total_assets: Total assets

    Returns:
        Investment rate (negative for factor)
    """
    if total_assets <= 0:
        return None
    return -capex / total_assets  # Negative so low capex = higher score


def net_operating_assets_growth(
    noa_current: float,
    noa_prior: float,
    total_assets: float
) -> Optional[float]:
    """
    Net Operating Assets Growth.

    Alternative investment factor.

    NOA = Operating Assets - Operating Liabilities

    Args:
        noa_current: Current NOA
        noa_prior: Prior NOA
        total_assets: Total assets for scaling

    Returns:
        NOA growth (negative for factor)
    """
    if total_assets <= 0:
        return None
    growth = (noa_current - noa_prior) / total_assets
    return -growth


# =============================================================================
# Quality Factors
# =============================================================================

def accruals_factor(
    accruals: float,
    total_assets: float
) -> Optional[float]:
    """
    Accruals Factor.

    Low accruals = high earnings quality = higher expected returns.

    Accruals = (ΔCA - ΔCash) - (ΔCL - ΔSTD) - Depreciation

    Args:
        accruals: Total accruals
        total_assets: Total assets

    Returns:
        Accruals ratio (negative so low accruals = high score)
    """
    if total_assets <= 0:
        return None
    return -accruals / total_assets


def earnings_quality_factor(
    cash_from_operations: float,
    net_income: float
) -> Optional[float]:
    """
    Earnings Quality Factor.

    CFO / Net Income

    Higher ratio = higher quality earnings.

    Args:
        cash_from_operations: Operating cash flow
        net_income: Net income

    Returns:
        Earnings quality ratio
    """
    if net_income == 0:
        return None
    # Handle negative earnings
    if net_income < 0:
        return cash_from_operations / abs(net_income) if cash_from_operations > 0 else 0
    return cash_from_operations / net_income


def earnings_stability_factor(
    earnings_std: float,
    mean_earnings: float
) -> Optional[float]:
    """
    Earnings Stability Factor.

    Lower volatility = higher quality = higher expected returns.

    CV = Std(Earnings) / Mean(Earnings)

    Args:
        earnings_std: Standard deviation of earnings
        mean_earnings: Mean earnings

    Returns:
        Negative coefficient of variation (for factor scoring)
    """
    if mean_earnings == 0:
        return None
    cv = earnings_std / abs(mean_earnings)
    return -cv  # Negative so stable = higher score


def leverage_quality_factor(
    debt: float,
    equity: float
) -> Optional[float]:
    """
    Leverage Quality Factor.

    Lower leverage = higher quality = higher expected returns.

    Args:
        debt: Total debt
        equity: Book equity

    Returns:
        Negative D/E (for factor scoring)
    """
    if equity <= 0:
        return None
    return -debt / equity


def composite_quality_score(
    gross_profitability: Optional[float],
    accruals_ratio: Optional[float],
    earnings_quality: Optional[float],
    leverage: Optional[float],
    weights: Optional[Dict[str, float]] = None
) -> Optional[float]:
    """
    Composite Quality Score.

    Combines profitability, accruals, earnings quality, and leverage.

    Args:
        gross_profitability: GP/A ratio
        accruals_ratio: Accruals/Assets
        earnings_quality: CFO/NI ratio
        leverage: D/E ratio (will be inverted)
        weights: Optional custom weights

    Returns:
        Composite quality score
    """
    if weights is None:
        weights = {
            'profitability': 0.35,
            'accruals': 0.25,
            'earnings_quality': 0.25,
            'leverage': 0.15,
        }

    score = 0.0
    total_weight = 0.0

    if gross_profitability is not None:
        score += weights['profitability'] * gross_profitability
        total_weight += weights['profitability']

    if accruals_ratio is not None:
        # Already inverted in accruals_factor
        score += weights['accruals'] * accruals_ratio
        total_weight += weights['accruals']

    if earnings_quality is not None:
        score += weights['earnings_quality'] * min(earnings_quality, 2.0)  # Cap at 2
        total_weight += weights['earnings_quality']

    if leverage is not None:
        # Invert so low leverage = high score
        score += weights['leverage'] * (-leverage)
        total_weight += weights['leverage']

    if total_weight <= 0:
        return None

    return score / total_weight


# =============================================================================
# Momentum Proxies
# =============================================================================

def earnings_momentum(
    earnings_change: float,
    prior_earnings: float
) -> Optional[float]:
    """
    Earnings Momentum (SUE - Standardized Unexpected Earnings).

    Proxy for price momentum using fundamentals.

    Args:
        earnings_change: Change in earnings (current - prior)
        prior_earnings: Prior period earnings

    Returns:
        Earnings surprise / change
    """
    if prior_earnings == 0:
        return None
    return earnings_change / abs(prior_earnings)


def revenue_momentum(
    revenue_growth: float
) -> float:
    """
    Revenue Momentum.

    Recent revenue growth as momentum proxy.

    Args:
        revenue_growth: YoY revenue growth rate

    Returns:
        Revenue momentum
    """
    return revenue_growth


def analyst_revision_proxy(
    current_eps_estimate: float,
    prior_eps_estimate: float
) -> Optional[float]:
    """
    Analyst Revision Proxy.

    Estimate revisions predict future returns.

    Args:
        current_eps_estimate: Current consensus EPS estimate
        prior_eps_estimate: Prior consensus EPS estimate

    Returns:
        Revision percentage
    """
    if prior_eps_estimate == 0:
        return None
    return (current_eps_estimate - prior_eps_estimate) / abs(prior_eps_estimate)


# =============================================================================
# Factor Score Calculation
# =============================================================================

@dataclass
class FactorExposures:
    """Company's exposures to various factors."""
    value: Optional[float] = None
    size: Optional[float] = None
    profitability: Optional[float] = None
    investment: Optional[float] = None
    quality: Optional[float] = None
    momentum: Optional[float] = None


def calculate_factor_exposures(
    company: CompanyFundamentals,
    prior_assets: float = 0.0,
    accruals: float = 0.0,
    earnings_change: float = 0.0,
    prior_earnings: float = 0.0
) -> FactorExposures:
    """
    Calculate Company's Factor Exposures.

    Args:
        company: Company fundamental data
        prior_assets: Prior period assets (for investment factor)
        accruals: Accruals (for quality factor)
        earnings_change: Earnings change (for momentum)
        prior_earnings: Prior earnings (for momentum)

    Returns:
        FactorExposures with all factor scores
    """
    # Value composite
    value = composite_value_score(
        company.book_to_market,
        company.earnings_yield,
        cash_flow_yield_factor(company.cash_flow, company.market_cap),
        sales_to_price_factor(company.sales, company.market_cap)
    )

    # Size
    size = size_factor(company.market_cap, log_transform=True)

    # Profitability
    profitability = company.gross_profitability

    # Investment
    investment = None
    if prior_assets > 0:
        investment = asset_growth_factor(company.total_assets, prior_assets)

    # Quality
    quality = composite_quality_score(
        profitability,
        accruals_factor(accruals, company.total_assets) if company.total_assets > 0 else None,
        earnings_quality_factor(company.cash_flow, company.net_income),
        company.debt_to_equity
    )

    # Momentum (fundamental-based)
    momentum = None
    if prior_earnings != 0:
        momentum = earnings_momentum(earnings_change, prior_earnings)

    return FactorExposures(
        value=value,
        size=size,
        profitability=profitability,
        investment=investment,
        quality=quality,
        momentum=momentum
    )


# =============================================================================
# Factor Analysis Class
# =============================================================================

class FactorAnalysis:
    """
    Factor Model Analysis.

    Calculates and analyzes factor exposures from fundamentals.
    """

    def __init__(self, companies: List[CompanyFundamentals]):
        """Initialize with list of companies."""
        self.companies = companies

    def calculate_all_exposures(
        self,
        prior_data: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, FactorExposures]:
        """
        Calculate factor exposures for all companies.

        Args:
            prior_data: Prior period data by ticker for momentum/investment

        Returns:
            Dictionary of exposures by ticker
        """
        exposures = {}

        for company in self.companies:
            prior = prior_data.get(company.ticker, {}) if prior_data else {}

            exposures[company.ticker] = calculate_factor_exposures(
                company,
                prior_assets=prior.get('total_assets', 0),
                accruals=prior.get('accruals', company.accruals),
                earnings_change=company.earnings - prior.get('earnings', company.earnings),
                prior_earnings=prior.get('earnings', 0)
            )

        return exposures

    def rank_by_factor(
        self,
        factor: str,
        ascending: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Rank companies by a specific factor.

        Args:
            factor: Factor name ('value', 'size', 'profitability', 'investment', 'quality', 'momentum')
            ascending: Sort ascending (True) or descending (False)

        Returns:
            List of (ticker, factor_value) tuples, sorted
        """
        exposures = self.calculate_all_exposures()

        # Extract factor values
        factor_values = []
        for ticker, exp in exposures.items():
            value = getattr(exp, factor, None)
            if value is not None:
                factor_values.append((ticker, value))

        # Sort
        factor_values.sort(key=lambda x: x[1], reverse=not ascending)

        return factor_values

    def quintile_portfolios(
        self,
        factor: str
    ) -> Dict[int, List[str]]:
        """
        Create quintile portfolios based on factor.

        Args:
            factor: Factor name

        Returns:
            Dictionary mapping quintile (1-5) to list of tickers
        """
        ranked = self.rank_by_factor(factor, ascending=True)

        if not ranked:
            return {}

        n = len(ranked)
        quintile_size = n // 5

        portfolios = {1: [], 2: [], 3: [], 4: [], 5: []}

        for i, (ticker, _) in enumerate(ranked):
            quintile = min(5, i // quintile_size + 1)
            portfolios[quintile].append(ticker)

        return portfolios

    def multi_factor_score(
        self,
        weights: Optional[Dict[str, float]] = None
    ) -> List[Tuple[str, float]]:
        """
        Calculate multi-factor composite score.

        Args:
            weights: Factor weights (default: equal weight)

        Returns:
            List of (ticker, composite_score) tuples, sorted
        """
        if weights is None:
            weights = {
                'value': 0.20,
                'size': 0.10,
                'profitability': 0.25,
                'investment': 0.15,
                'quality': 0.20,
                'momentum': 0.10,
            }

        exposures = self.calculate_all_exposures()

        # Normalize factors to z-scores
        # First collect all values
        factor_values = {f: [] for f in weights.keys()}
        for exp in exposures.values():
            for factor in weights.keys():
                val = getattr(exp, factor, None)
                if val is not None:
                    factor_values[factor].append(val)

        # Calculate means and stds
        factor_stats = {}
        for factor, values in factor_values.items():
            if values:
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                std = math.sqrt(variance) if variance > 0 else 1
                factor_stats[factor] = (mean, std)
            else:
                factor_stats[factor] = (0, 1)

        # Calculate composite scores
        scores = []
        for ticker, exp in exposures.items():
            composite = 0.0
            total_weight = 0.0

            for factor, weight in weights.items():
                val = getattr(exp, factor, None)
                if val is not None:
                    mean, std = factor_stats[factor]
                    z_score = (val - mean) / std if std > 0 else 0
                    composite += weight * z_score
                    total_weight += weight

            if total_weight > 0:
                composite /= total_weight
                scores.append((ticker, composite))

        # Sort by composite score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores

    def factor_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate correlation between factors.

        Returns:
            Correlation matrix as nested dictionary
        """
        exposures = self.calculate_all_exposures()
        factors = ['value', 'size', 'profitability', 'investment', 'quality', 'momentum']

        # Collect factor values
        factor_data = {f: [] for f in factors}
        for exp in exposures.values():
            for factor in factors:
                val = getattr(exp, factor, None)
                factor_data[factor].append(val)

        # Calculate correlations
        correlations = {f: {} for f in factors}

        for f1 in factors:
            for f2 in factors:
                # Get paired observations
                pairs = [(factor_data[f1][i], factor_data[f2][i])
                         for i in range(len(factor_data[f1]))
                         if factor_data[f1][i] is not None and factor_data[f2][i] is not None]

                if len(pairs) < 3:
                    correlations[f1][f2] = None
                    continue

                x = [p[0] for p in pairs]
                y = [p[1] for p in pairs]

                mean_x = sum(x) / len(x)
                mean_y = sum(y) / len(y)

                cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))) / len(x)
                var_x = sum((xi - mean_x) ** 2 for xi in x) / len(x)
                var_y = sum((yi - mean_y) ** 2 for yi in y) / len(y)

                if var_x > 0 and var_y > 0:
                    correlations[f1][f2] = cov / math.sqrt(var_x * var_y)
                else:
                    correlations[f1][f2] = 0

        return correlations
