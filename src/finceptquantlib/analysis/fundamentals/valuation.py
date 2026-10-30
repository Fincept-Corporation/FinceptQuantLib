"""
Valuation Ratios

Comprehensive valuation analysis with 20+ ratios:
- Price Multiples (P/E, P/B, P/S, P/CF)
- Enterprise Value Multiples (EV/EBITDA, EV/EBIT, EV/Revenue)
- Yield Metrics (Earnings Yield, FCF Yield, Dividend Yield)
- Value Investing Metrics (Graham Number, NCAV, Margin of Safety)

All ratios help assess whether a company is over or undervalued.
"""

from dataclasses import dataclass
from typing import Optional, Dict
import math
from .statements import (
    IncomeStatement, BalanceSheet, CashFlowStatement, FinancialStatements
)


@dataclass
class ValuationResult:
    """Result of valuation analysis."""
    # Price Multiples
    price_to_earnings: float = 0.0
    price_to_earnings_forward: float = 0.0
    price_to_book: float = 0.0
    price_to_tangible_book: float = 0.0
    price_to_sales: float = 0.0
    price_to_cash_flow: float = 0.0
    price_to_free_cash_flow: float = 0.0

    # EV Multiples
    ev_to_ebitda: float = 0.0
    ev_to_ebit: float = 0.0
    ev_to_revenue: float = 0.0
    ev_to_fcf: float = 0.0
    ev_to_invested_capital: float = 0.0

    # Yield Metrics
    earnings_yield: float = 0.0
    fcf_yield: float = 0.0
    dividend_yield: float = 0.0

    # Value Investing
    peg_ratio: float = 0.0
    graham_number: float = 0.0
    ncav_per_share: float = 0.0
    margin_of_safety: float = 0.0
    tobins_q: float = 0.0


@dataclass
class MarketData:
    """Market data required for valuation."""
    stock_price: float = 0.0
    shares_outstanding: float = 0.0
    forward_eps: Optional[float] = None
    expected_growth_rate: Optional[float] = None

    @property
    def market_cap(self) -> float:
        """Market capitalization = Price × Shares Outstanding"""
        return self.stock_price * self.shares_outstanding


def enterprise_value(
    market: MarketData,
    balance: BalanceSheet,
    minority_interest: float = 0.0,
    preferred_stock: float = 0.0
) -> float:
    """
    Enterprise Value = Market Cap + Total Debt + Minority Interest + Preferred - Cash

    Represents the total value of the business to all capital providers.

    Args:
        market: Market data with stock price and shares
        balance: Balance sheet
        minority_interest: Value of minority interest (if any)
        preferred_stock: Value of preferred stock (if any)

    Returns: Enterprise value
    """
    return (
        market.market_cap +
        balance.total_debt +
        minority_interest +
        preferred_stock -
        balance.cash_and_equivalents
    )


# =============================================================================
# PRICE MULTIPLES
# =============================================================================

def price_to_earnings(
    market: MarketData,
    income: IncomeStatement
) -> float:
    """
    Price to Earnings = Stock Price / Earnings Per Share
    Also known as: P/E Ratio, Earnings Multiple

    The most common valuation metric.

    Benchmark:
    - <10: May be undervalued or distressed
    - 10-20: Typical for mature companies
    - 20-30: Growth expectations priced in
    - >30: High growth or possibly overvalued

    Note: Meaningless when EPS is negative
    """
    if income.eps_basic <= 0:
        return float('inf') if market.stock_price > 0 else 0.0
    return market.stock_price / income.eps_basic


def price_to_earnings_diluted(
    market: MarketData,
    income: IncomeStatement
) -> float:
    """
    P/E (Diluted) = Stock Price / Diluted EPS

    Uses diluted EPS which accounts for stock options, convertibles, etc.
    More conservative than basic P/E.
    """
    if income.eps_diluted <= 0:
        return float('inf') if market.stock_price > 0 else 0.0
    return market.stock_price / income.eps_diluted


def price_to_earnings_forward(
    market: MarketData,
    forward_eps: float
) -> float:
    """
    Forward P/E = Stock Price / Forward EPS Estimate

    Uses analyst estimates of next year's earnings.
    More forward-looking than trailing P/E.

    Args:
        market: Market data
        forward_eps: Consensus forward EPS estimate
    """
    if forward_eps <= 0:
        return float('inf') if market.stock_price > 0 else 0.0
    return market.stock_price / forward_eps


def price_to_book(
    market: MarketData,
    balance: BalanceSheet
) -> float:
    """
    Price to Book = Stock Price / Book Value Per Share
    Book Value Per Share = Total Equity / Shares Outstanding

    Compares market value to accounting book value.

    Benchmark:
    - <1.0: Trading below book value (may be undervalued)
    - 1.0-3.0: Typical
    - >3.0: Premium for growth or intangible value
    """
    if balance.total_equity <= 0:
        return float('inf') if market.stock_price > 0 else 0.0
    book_value_per_share = balance.total_equity / market.shares_outstanding
    return market.stock_price / book_value_per_share


def price_to_tangible_book(
    market: MarketData,
    balance: BalanceSheet
) -> float:
    """
    Price to Tangible Book = Price / Tangible Book Value Per Share

    Tangible Book = Equity - Goodwill - Intangibles

    More conservative than P/B; relevant for acquisitive companies.
    """
    tangible_equity = balance.total_equity - balance.goodwill - balance.intangible_assets
    if tangible_equity <= 0:
        return float('inf') if market.stock_price > 0 else 0.0
    tangible_book_per_share = tangible_equity / market.shares_outstanding
    return market.stock_price / tangible_book_per_share


def price_to_sales(
    market: MarketData,
    income: IncomeStatement
) -> float:
    """
    Price to Sales = Market Cap / Revenue
    Also: Stock Price / Revenue Per Share

    Useful when earnings are negative or volatile.

    Benchmark:
    - <1.0: May be undervalued
    - 1.0-3.0: Typical
    - >5.0: High growth expectations
    """
    if income.revenue <= 0:
        return float('inf') if market.stock_price > 0 else 0.0
    return market.market_cap / income.revenue


def price_to_cash_flow(
    market: MarketData,
    cash_flow: CashFlowStatement
) -> float:
    """
    Price to Cash Flow = Market Cap / Operating Cash Flow
    Also: Stock Price / Cash Flow Per Share

    Cash-based alternative to P/E.

    Benchmark:
    - <10: Potentially undervalued
    - 10-20: Typical
    - >20: Premium valuation
    """
    if cash_flow.cash_from_operations <= 0:
        return float('inf') if market.stock_price > 0 else 0.0
    return market.market_cap / cash_flow.cash_from_operations


def price_to_free_cash_flow(
    market: MarketData,
    cash_flow: CashFlowStatement
) -> float:
    """
    Price to Free Cash Flow = Market Cap / Free Cash Flow

    FCF = Operating Cash Flow - CapEx

    Most relevant for mature, cash-generating businesses.

    Benchmark:
    - <15: Potentially undervalued
    - 15-25: Typical
    - >25: Premium valuation or high growth
    """
    fcf = cash_flow.free_cash_flow
    if fcf <= 0:
        return float('inf') if market.stock_price > 0 else 0.0
    return market.market_cap / fcf


# =============================================================================
# ENTERPRISE VALUE MULTIPLES
# =============================================================================

def ev_to_ebitda(
    market: MarketData,
    income: IncomeStatement,
    balance: BalanceSheet
) -> float:
    """
    EV/EBITDA = Enterprise Value / EBITDA

    Most widely used EV multiple. Capital structure neutral.

    Benchmark:
    - <6x: May be undervalued
    - 6-10x: Typical for mature companies
    - 10-15x: Growth premium
    - >15x: High growth or possibly overvalued
    """
    ev = enterprise_value(market, balance)
    if income.ebitda <= 0:
        return float('inf') if ev > 0 else 0.0
    return ev / income.ebitda


def ev_to_ebit(
    market: MarketData,
    income: IncomeStatement,
    balance: BalanceSheet
) -> float:
    """
    EV/EBIT = Enterprise Value / EBIT (Operating Income)

    More conservative than EV/EBITDA; includes depreciation.
    Better for capital-intensive businesses.

    Benchmark: Typically 1-3x higher than EV/EBITDA
    """
    ev = enterprise_value(market, balance)
    if income.operating_income <= 0:
        return float('inf') if ev > 0 else 0.0
    return ev / income.operating_income


def ev_to_revenue(
    market: MarketData,
    income: IncomeStatement,
    balance: BalanceSheet
) -> float:
    """
    EV/Revenue = Enterprise Value / Revenue
    Also known as: EV/Sales

    Useful for unprofitable companies or comparing across industries.

    Benchmark:
    - <1x: May be undervalued
    - 1-3x: Typical
    - >5x: High growth SaaS-like expectations
    """
    ev = enterprise_value(market, balance)
    if income.revenue <= 0:
        return float('inf') if ev > 0 else 0.0
    return ev / income.revenue


def ev_to_fcf(
    market: MarketData,
    cash_flow: CashFlowStatement,
    balance: BalanceSheet
) -> float:
    """
    EV/FCF = Enterprise Value / Free Cash Flow

    Cash-based EV multiple. Useful for mature businesses.
    """
    ev = enterprise_value(market, balance)
    fcf = cash_flow.free_cash_flow
    if fcf <= 0:
        return float('inf') if ev > 0 else 0.0
    return ev / fcf


def ev_to_invested_capital(
    market: MarketData,
    balance: BalanceSheet
) -> float:
    """
    EV/Invested Capital = Enterprise Value / Invested Capital

    Measures market value relative to capital invested.

    >1.0: Market believes company creates value
    <1.0: Market believes company destroys value
    """
    ev = enterprise_value(market, balance)
    if balance.invested_capital <= 0:
        return float('inf') if ev > 0 else 0.0
    return ev / balance.invested_capital


# =============================================================================
# YIELD METRICS
# =============================================================================

def earnings_yield(
    market: MarketData,
    income: IncomeStatement
) -> float:
    """
    Earnings Yield = EPS / Stock Price = 1 / P/E

    Inverse of P/E ratio. Can be compared to bond yields.

    Benchmark: Higher is better; compare to risk-free rate
    """
    if market.stock_price <= 0:
        return 0.0
    return income.eps_basic / market.stock_price


def fcf_yield(
    market: MarketData,
    cash_flow: CashFlowStatement
) -> float:
    """
    FCF Yield = Free Cash Flow / Market Cap

    Cash-based yield metric. Important for value investors.

    Benchmark:
    - >10%: Potentially attractive
    - 5-10%: Reasonable
    - <5%: Premium valuation
    """
    if market.market_cap <= 0:
        return 0.0
    return cash_flow.free_cash_flow / market.market_cap


def dividend_yield(
    market: MarketData,
    income: IncomeStatement
) -> float:
    """
    Dividend Yield = Dividend Per Share / Stock Price

    Annual dividend return on investment.

    Benchmark:
    - 0%: No dividend
    - 2-4%: Typical for dividend payers
    - >5%: High yield (check sustainability)
    """
    if market.stock_price <= 0:
        return 0.0
    return income.dividends_per_share / market.stock_price


def shareholder_yield(
    market: MarketData,
    income: IncomeStatement,
    cash_flow: CashFlowStatement
) -> float:
    """
    Shareholder Yield = Dividend Yield + Buyback Yield

    Total cash return to shareholders including buybacks.
    """
    if market.market_cap <= 0:
        return 0.0

    div_yield = income.dividends_per_share / market.stock_price if market.stock_price > 0 else 0.0
    buyback_yield = cash_flow.shares_repurchased / market.market_cap

    return div_yield + buyback_yield


# =============================================================================
# VALUE INVESTING METRICS
# =============================================================================

def peg_ratio(
    market: MarketData,
    income: IncomeStatement,
    growth_rate: float
) -> float:
    """
    PEG Ratio = P/E / Earnings Growth Rate

    Adjusts P/E for growth. Popularized by Peter Lynch.

    Args:
        market: Market data
        income: Income statement
        growth_rate: Expected earnings growth rate (as decimal, e.g., 0.15 for 15%)

    Benchmark:
    - <1.0: Potentially undervalued relative to growth
    - 1.0: Fair value (Peter Lynch's rule)
    - >1.0: May be overvalued relative to growth
    """
    pe = price_to_earnings(market, income)
    if pe == float('inf') or growth_rate <= 0:
        return float('inf')

    # Convert growth rate to percentage for PEG calculation
    return pe / (growth_rate * 100)


def graham_number(
    income: IncomeStatement,
    balance: BalanceSheet
) -> float:
    """
    Graham Number = √(22.5 × EPS × Book Value Per Share)

    Benjamin Graham's formula for maximum price to pay.

    Returns the maximum price a defensive investor should pay.
    """
    if income.eps_basic <= 0 or balance.total_equity <= 0:
        return 0.0

    bvps = balance.total_equity / income.shares_outstanding
    return math.sqrt(22.5 * income.eps_basic * bvps)


def ncav_per_share(
    balance: BalanceSheet,
    shares_outstanding: float
) -> float:
    """
    Net Current Asset Value Per Share = (Current Assets - Total Liabilities) / Shares

    Graham's "net-net" formula. Very conservative valuation floor.

    If stock trades below NCAV, may be deep value opportunity.
    """
    if shares_outstanding <= 0:
        return 0.0
    ncav = balance.total_current_assets - balance.total_liabilities
    return ncav / shares_outstanding


def margin_of_safety(
    market: MarketData,
    intrinsic_value: float
) -> float:
    """
    Margin of Safety = (Intrinsic Value - Market Price) / Intrinsic Value

    Measures discount to intrinsic value.

    Args:
        market: Market data with current price
        intrinsic_value: Estimated intrinsic value per share

    Returns: Margin of safety as decimal (e.g., 0.3 for 30%)

    Benchmark:
    - >30%: Strong margin of safety
    - 15-30%: Moderate margin
    - <15%: Limited margin for error
    - Negative: Trading above intrinsic value
    """
    if intrinsic_value <= 0:
        return 0.0
    return (intrinsic_value - market.stock_price) / intrinsic_value


def tobins_q(
    market: MarketData,
    balance: BalanceSheet
) -> float:
    """
    Tobin's Q = Market Value of Assets / Replacement Cost of Assets

    Approximation: (Market Cap + Liabilities) / Total Assets

    Interpretation:
    - >1.0: Market values company above replacement cost
    - <1.0: Cheaper to buy company than build from scratch
    """
    if balance.total_assets <= 0:
        return 0.0
    market_value = market.market_cap + balance.total_liabilities
    return market_value / balance.total_assets


# =============================================================================
# COMPREHENSIVE ANALYSIS CLASS
# =============================================================================

class ValuationAnalysis:
    """
    Comprehensive valuation analysis.

    Calculates all valuation ratios for a company.

    Example:
        market = MarketData(stock_price=150.0, shares_outstanding=1e9)
        analysis = ValuationAnalysis(statements, market)
        result = analysis.analyze()
        print(f"P/E: {result.price_to_earnings:.1f}x")
        print(f"EV/EBITDA: {result.ev_to_ebitda:.1f}x")
    """

    def __init__(
        self,
        current: FinancialStatements,
        market: MarketData,
        prior: Optional[FinancialStatements] = None
    ):
        self.current = current
        self.market = market
        self.prior = prior

    def analyze(self) -> ValuationResult:
        """Calculate all valuation ratios."""
        income = self.current.income_statement
        balance = self.current.balance_sheet
        cash_flow_stmt = self.current.cash_flow
        market = self.market

        # Calculate growth rate for PEG if prior data available
        growth_rate = 0.10  # Default 10%
        if self.prior:
            prior_eps = self.prior.income_statement.eps_basic
            if prior_eps > 0 and income.eps_basic > 0:
                growth_rate = (income.eps_basic - prior_eps) / prior_eps

        # Graham number
        graham = graham_number(income, balance)

        # Forward P/E if available
        forward_pe = 0.0
        if market.forward_eps and market.forward_eps > 0:
            forward_pe = price_to_earnings_forward(market, market.forward_eps)

        return ValuationResult(
            # Price Multiples
            price_to_earnings=price_to_earnings(market, income),
            price_to_earnings_forward=forward_pe,
            price_to_book=price_to_book(market, balance),
            price_to_tangible_book=price_to_tangible_book(market, balance),
            price_to_sales=price_to_sales(market, income),
            price_to_cash_flow=price_to_cash_flow(market, cash_flow_stmt),
            price_to_free_cash_flow=price_to_free_cash_flow(market, cash_flow_stmt),

            # EV Multiples
            ev_to_ebitda=ev_to_ebitda(market, income, balance),
            ev_to_ebit=ev_to_ebit(market, income, balance),
            ev_to_revenue=ev_to_revenue(market, income, balance),
            ev_to_fcf=ev_to_fcf(market, cash_flow_stmt, balance),
            ev_to_invested_capital=ev_to_invested_capital(market, balance),

            # Yields
            earnings_yield=earnings_yield(market, income),
            fcf_yield=fcf_yield(market, cash_flow_stmt),
            dividend_yield=dividend_yield(market, income),

            # Value Investing
            peg_ratio=peg_ratio(market, income, growth_rate) if growth_rate > 0 else 0.0,
            graham_number=graham,
            ncav_per_share=ncav_per_share(balance, market.shares_outstanding),
            margin_of_safety=margin_of_safety(market, graham),
            tobins_q=tobins_q(market, balance),
        )

    def to_dict(self) -> Dict[str, float]:
        """Return all ratios as dictionary."""
        result = self.analyze()
        return {
            'price_to_earnings': result.price_to_earnings,
            'price_to_book': result.price_to_book,
            'price_to_sales': result.price_to_sales,
            'price_to_cash_flow': result.price_to_cash_flow,
            'price_to_free_cash_flow': result.price_to_free_cash_flow,
            'ev_to_ebitda': result.ev_to_ebitda,
            'ev_to_ebit': result.ev_to_ebit,
            'ev_to_revenue': result.ev_to_revenue,
            'earnings_yield': result.earnings_yield,
            'fcf_yield': result.fcf_yield,
            'dividend_yield': result.dividend_yield,
            'peg_ratio': result.peg_ratio,
            'graham_number': result.graham_number,
        }

    def assess_valuation(self) -> str:
        """Provide a qualitative assessment of valuation."""
        result = self.analyze()
        market = self.market

        cheap_signals = []
        expensive_signals = []

        # P/E Assessment
        if result.price_to_earnings < 10:
            cheap_signals.append("Low P/E (<10x)")
        elif result.price_to_earnings > 25:
            expensive_signals.append("High P/E (>25x)")

        # P/B Assessment
        if result.price_to_book < 1.0:
            cheap_signals.append("Trading below book value")
        elif result.price_to_book > 4.0:
            expensive_signals.append("High P/B (>4x)")

        # EV/EBITDA Assessment
        if result.ev_to_ebitda < 6:
            cheap_signals.append("Low EV/EBITDA (<6x)")
        elif result.ev_to_ebitda > 12:
            expensive_signals.append("High EV/EBITDA (>12x)")

        # FCF Yield Assessment
        if result.fcf_yield > 0.10:
            cheap_signals.append("High FCF yield (>10%)")
        elif result.fcf_yield < 0.03 and result.fcf_yield > 0:
            expensive_signals.append("Low FCF yield (<3%)")

        # Graham Number Assessment
        if result.graham_number > 0 and market.stock_price < result.graham_number:
            cheap_signals.append(f"Below Graham Number (${result.graham_number:.2f})")
        elif result.graham_number > 0 and market.stock_price > result.graham_number * 1.5:
            expensive_signals.append("Well above Graham Number")

        # NCAV Assessment
        if result.ncav_per_share > 0 and market.stock_price < result.ncav_per_share:
            cheap_signals.append("Net-net: Trading below NCAV")

        assessment = "VALUATION ASSESSMENT\n" + "=" * 40 + "\n\n"
        assessment += f"Current Price: ${market.stock_price:.2f}\n"
        assessment += f"Market Cap: ${market.market_cap/1e9:.2f}B\n\n"

        if cheap_signals:
            assessment += "VALUE SIGNALS:\n"
            for s in cheap_signals:
                assessment += f"  + {s}\n"
            assessment += "\n"

        if expensive_signals:
            assessment += "PREMIUM SIGNALS:\n"
            for s in expensive_signals:
                assessment += f"  - {s}\n"

        score = len(cheap_signals) - len(expensive_signals)
        if score >= 2:
            assessment += "\nOVERALL: Potentially UNDERVALUED\n"
        elif score <= -2:
            assessment += "\nOVERALL: Potentially OVERVALUED\n"
        else:
            assessment += "\nOVERALL: FAIRLY VALUED\n"

        return assessment
