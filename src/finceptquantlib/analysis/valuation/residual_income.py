"""
Residual Income and Economic Value Added (EVA) Analysis

Valuation and performance metrics based on economic profit:
- Economic Value Added (EVA)
- Market Value Added (MVA)
- Cash Value Added (CVA)
- Residual Income Valuation Model
- Economic Profit Margin
- Spread Analysis (ROIC - WACC)
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


# =============================================================================
# Economic Value Added (EVA)
# =============================================================================

def economic_value_added(
    nopat: float,
    invested_capital: float,
    wacc: float
) -> float:
    """
    Economic Value Added (EVA).

    Measures true economic profit after cost of capital.

    EVA = NOPAT - (Invested Capital × WACC)
    EVA = (ROIC - WACC) × Invested Capital

    Positive EVA = Creating shareholder value
    Negative EVA = Destroying shareholder value

    Args:
        nopat: Net Operating Profit After Tax
        invested_capital: Total invested capital (debt + equity)
        wacc: Weighted average cost of capital

    Returns:
        EVA amount
    """
    capital_charge = invested_capital * wacc
    return nopat - capital_charge


def eva_spread(roic: float, wacc: float) -> float:
    """
    EVA Spread.

    Difference between return on invested capital and cost of capital.

    Spread = ROIC - WACC

    Positive spread = Value creation
    Negative spread = Value destruction

    Args:
        roic: Return on invested capital
        wacc: Weighted average cost of capital

    Returns:
        Spread as decimal
    """
    return roic - wacc


def nopat(
    operating_income: float,
    tax_rate: float,
    adjustments: float = 0.0
) -> float:
    """
    Net Operating Profit After Tax (NOPAT).

    NOPAT = Operating Income × (1 - Tax Rate) + Adjustments

    Common adjustments:
    - Add back: Goodwill amortization, R&D capitalization
    - Remove: Non-operating income, unusual items

    Args:
        operating_income: EBIT or operating income
        tax_rate: Effective tax rate
        adjustments: After-tax adjustments

    Returns:
        NOPAT amount
    """
    return operating_income * (1 - tax_rate) + adjustments


def invested_capital(
    total_assets: float,
    non_interest_bearing_liabilities: float,
    excess_cash: float = 0.0,
    non_operating_assets: float = 0.0
) -> float:
    """
    Invested Capital.

    Capital employed in core operations.

    Invested Capital = Total Assets - NIBLs - Excess Cash - Non-Operating Assets

    Alternative: Debt + Equity - Excess Cash

    Args:
        total_assets: Total assets
        non_interest_bearing_liabilities: AP, accrued expenses, deferred revenue
        excess_cash: Cash beyond operating needs
        non_operating_assets: Non-core assets

    Returns:
        Invested capital amount
    """
    return total_assets - non_interest_bearing_liabilities - excess_cash - non_operating_assets


def eva_margin(eva: float, revenue: float) -> Optional[float]:
    """
    EVA Margin.

    Economic profit as percentage of revenue.

    Args:
        eva: Economic value added
        revenue: Total revenue

    Returns:
        EVA margin as decimal
    """
    if revenue <= 0:
        return None
    return eva / revenue


# =============================================================================
# Market Value Added (MVA)
# =============================================================================

def market_value_added(
    market_value: float,
    invested_capital: float
) -> float:
    """
    Market Value Added (MVA).

    Difference between market value and capital invested.

    MVA = Market Value of Firm - Invested Capital

    Represents cumulative wealth created for shareholders.

    Args:
        market_value: Market cap + market value of debt
        invested_capital: Book value of invested capital

    Returns:
        MVA amount
    """
    return market_value - invested_capital


def mva_from_eva(
    eva_stream: List[float],
    wacc: float,
    terminal_growth: float = 0.0
) -> float:
    """
    MVA from Present Value of Future EVA.

    MVA = PV of all future EVAs

    Args:
        eva_stream: Projected EVA for each year
        wacc: Discount rate
        terminal_growth: Terminal growth rate for EVA

    Returns:
        MVA as PV of EVA stream
    """
    if not eva_stream:
        return 0.0

    pv = 0.0
    for t, eva in enumerate(eva_stream, 1):
        pv += eva / ((1 + wacc) ** t)

    # Terminal value if growth provided
    if terminal_growth > 0 and terminal_growth < wacc:
        last_eva = eva_stream[-1]
        terminal_eva = last_eva * (1 + terminal_growth)
        terminal_value = terminal_eva / (wacc - terminal_growth)
        pv += terminal_value / ((1 + wacc) ** len(eva_stream))

    return pv


# =============================================================================
# Cash Value Added (CVA)
# =============================================================================

def cash_value_added(
    operating_cash_flow: float,
    economic_depreciation: float,
    capital_charge: float
) -> float:
    """
    Cash Value Added (CVA).

    Cash-based economic profit measure.

    CVA = Operating Cash Flow - Economic Depreciation - Capital Charge

    Economic depreciation = amount needed to replace assets at current cost.

    Args:
        operating_cash_flow: Cash from operations (adjusted)
        economic_depreciation: Replacement cost depreciation
        capital_charge: Invested capital × WACC

    Returns:
        CVA amount
    """
    return operating_cash_flow - economic_depreciation - capital_charge


def economic_depreciation(
    gross_investment: float,
    asset_life: int,
    wacc: float
) -> float:
    """
    Economic Depreciation (Sinking Fund Method).

    Amount to set aside annually to replace asset at end of life.

    ED = Gross Investment × [WACC / ((1 + WACC)^n - 1)]

    This is lower than straight-line in early years.

    Args:
        gross_investment: Original asset cost
        asset_life: Expected useful life in years
        wacc: Cost of capital

    Returns:
        Annual economic depreciation
    """
    if asset_life <= 0 or wacc <= 0:
        return gross_investment / max(asset_life, 1)

    factor = wacc / (((1 + wacc) ** asset_life) - 1)
    return gross_investment * factor


def cfroi(
    gross_cash_flow: float,
    gross_investment: float,
    asset_life: int,
    terminal_value: float = 0.0
) -> Optional[float]:
    """
    Cash Flow Return on Investment (CFROI).

    Internal rate of return on gross investment.

    Solves for r in:
    Gross Investment = Σ(Gross Cash Flow / (1+r)^t) + Terminal Value / (1+r)^n

    Simplified approximation when terminal_value = 0:
    CFROI ≈ Gross Cash Flow / Gross Investment (adjusted for asset life)

    Args:
        gross_cash_flow: Annual gross cash flow
        gross_investment: Gross investment (inflation-adjusted)
        asset_life: Average asset life
        terminal_value: Residual/salvage value

    Returns:
        CFROI as decimal
    """
    if gross_investment <= 0:
        return None

    # Newton-Raphson to solve for IRR
    def npv(r: float) -> float:
        if r <= -1:
            return float('inf')
        pv = sum(gross_cash_flow / ((1 + r) ** t) for t in range(1, asset_life + 1))
        pv += terminal_value / ((1 + r) ** asset_life)
        return pv - gross_investment

    # Initial guess
    r = gross_cash_flow / gross_investment

    for _ in range(50):
        f = npv(r)
        if abs(f) < 0.0001:
            return r

        # Numerical derivative
        h = 0.0001
        f_prime = (npv(r + h) - f) / h

        if abs(f_prime) < 1e-10:
            break

        r = r - f / f_prime

        # Bounds
        r = max(-0.99, min(r, 2.0))

    return r


# =============================================================================
# Residual Income Valuation
# =============================================================================

def residual_income(
    net_income: float,
    equity: float,
    cost_of_equity: float
) -> float:
    """
    Residual Income (RI).

    Profit remaining after equity charge.

    RI = Net Income - (Equity × Cost of Equity)
    RI = (ROE - Cost of Equity) × Equity

    Args:
        net_income: Net income attributable to equity
        equity: Book value of equity
        cost_of_equity: Required return on equity

    Returns:
        Residual income amount
    """
    equity_charge = equity * cost_of_equity
    return net_income - equity_charge


def residual_income_valuation(
    book_value: float,
    ri_forecasts: List[float],
    cost_of_equity: float,
    terminal_growth: float = 0.0,
    persistence_factor: float = 1.0
) -> float:
    """
    Residual Income Valuation Model.

    Intrinsic Value = Book Value + PV of Future Residual Income

    V = BV + Σ(RI_t / (1+r)^t) + Terminal Value

    Args:
        book_value: Current book value of equity
        ri_forecasts: Projected residual income for each year
        cost_of_equity: Required return on equity
        terminal_growth: Long-term RI growth rate
        persistence_factor: RI persistence (0-1, where 1 = perpetual)

    Returns:
        Intrinsic value per share
    """
    if not ri_forecasts:
        return book_value

    # Present value of forecast period RI
    pv_ri = 0.0
    for t, ri in enumerate(ri_forecasts, 1):
        pv_ri += ri / ((1 + cost_of_equity) ** t)

    # Terminal value
    terminal_value = 0.0
    if persistence_factor > 0 and len(ri_forecasts) > 0:
        last_ri = ri_forecasts[-1]
        n = len(ri_forecasts)

        if terminal_growth > 0 and terminal_growth < cost_of_equity:
            # Growing perpetuity
            terminal_ri = last_ri * (1 + terminal_growth) * persistence_factor
            terminal_value = terminal_ri / (cost_of_equity - terminal_growth)
        else:
            # Fade to zero (convergence)
            # RI fades as ROE converges to cost of equity
            terminal_value = last_ri * persistence_factor / cost_of_equity

        terminal_value = terminal_value / ((1 + cost_of_equity) ** n)

    return book_value + pv_ri + terminal_value


def clean_surplus_ri(
    beginning_equity: float,
    net_income: float,
    dividends: float,
    other_comprehensive_income: float = 0.0
) -> Tuple[float, float]:
    """
    Clean Surplus Relation Check.

    Verifies: Ending Equity = Beginning Equity + NI - Dividends + OCI

    Clean surplus is required for RI valuation to work properly.

    Args:
        beginning_equity: Beginning book value
        net_income: Net income for period
        dividends: Dividends paid
        other_comprehensive_income: OCI items

    Returns:
        Tuple of (ending_equity, residual_income)
    """
    ending_equity = beginning_equity + net_income - dividends + other_comprehensive_income
    return ending_equity, net_income - dividends


# =============================================================================
# Multi-Stage Residual Income Model
# =============================================================================

def two_stage_ri_valuation(
    book_value: float,
    eps_forecast: List[float],
    dividend_payout: float,
    cost_of_equity: float,
    terminal_roe: float,
    terminal_growth: float
) -> Dict[str, float]:
    """
    Two-Stage Residual Income Valuation.

    Stage 1: Explicit forecast period with varying RI
    Stage 2: Terminal value with ROE converging to required return

    Args:
        book_value: Current book value per share
        eps_forecast: EPS forecasts for explicit period
        dividend_payout: Dividend payout ratio
        cost_of_equity: Required return on equity
        terminal_roe: Long-term sustainable ROE
        terminal_growth: Long-term growth rate

    Returns:
        Dictionary with valuation components
    """
    # Build equity and RI forecasts
    equity = book_value
    ri_forecasts = []
    equity_forecasts = [book_value]

    for eps in eps_forecast:
        # Residual income = EPS - equity charge
        ri = eps - (equity * cost_of_equity)
        ri_forecasts.append(ri)

        # Update equity (clean surplus)
        dividends = eps * dividend_payout
        equity = equity + eps - dividends
        equity_forecasts.append(equity)

    # Present value of explicit period RI
    pv_ri_stage1 = 0.0
    for t, ri in enumerate(ri_forecasts, 1):
        pv_ri_stage1 += ri / ((1 + cost_of_equity) ** t)

    # Terminal value
    n = len(eps_forecast)
    terminal_equity = equity_forecasts[-1]

    # Terminal RI based on sustainable ROE
    terminal_ri = terminal_equity * (terminal_roe - cost_of_equity)

    if terminal_growth < cost_of_equity:
        terminal_value = terminal_ri * (1 + terminal_growth) / (cost_of_equity - terminal_growth)
    else:
        terminal_value = terminal_ri / cost_of_equity

    pv_terminal = terminal_value / ((1 + cost_of_equity) ** n)

    intrinsic_value = book_value + pv_ri_stage1 + pv_terminal

    return {
        'intrinsic_value': intrinsic_value,
        'book_value': book_value,
        'pv_stage1_ri': pv_ri_stage1,
        'pv_terminal_value': pv_terminal,
        'terminal_equity': terminal_equity,
        'ri_forecasts': ri_forecasts,
    }


def justified_pb_ratio(
    roe: float,
    cost_of_equity: float,
    growth_rate: float
) -> Optional[float]:
    """
    Justified Price-to-Book Ratio.

    Derived from residual income model.

    P/B = (ROE - g) / (r - g)

    Where:
    - If ROE > r: P/B > 1 (premium to book)
    - If ROE < r: P/B < 1 (discount to book)
    - If ROE = r: P/B = 1

    Args:
        roe: Return on equity
        cost_of_equity: Required return on equity
        growth_rate: Sustainable growth rate

    Returns:
        Justified P/B ratio
    """
    if cost_of_equity <= growth_rate:
        return None

    return (roe - growth_rate) / (cost_of_equity - growth_rate)


# =============================================================================
# EVA Analysis Class
# =============================================================================

@dataclass
class EVAResult:
    """EVA calculation result."""
    eva: float
    nopat: float
    invested_capital: float
    wacc: float
    capital_charge: float
    eva_spread: float
    eva_margin: Optional[float] = None


@dataclass
class RIValuationResult:
    """Residual income valuation result."""
    intrinsic_value: float
    book_value: float
    pv_residual_income: float
    terminal_value: float
    premium_to_book: float


class EVAAnalysis:
    """
    Economic Value Added Analysis.

    Comprehensive EVA and residual income analysis.
    """

    def __init__(
        self,
        operating_income: float,
        tax_rate: float,
        total_assets: float,
        current_liabilities: float,
        excess_cash: float = 0.0,
        debt: float = 0.0,
        equity: float = 0.0,
        cost_of_debt: float = 0.0,
        cost_of_equity: float = 0.0,
        revenue: float = 0.0
    ):
        """Initialize EVA analysis."""
        self.operating_income = operating_income
        self.tax_rate = tax_rate
        self.total_assets = total_assets
        self.current_liabilities = current_liabilities
        self.excess_cash = excess_cash
        self.debt = debt
        self.equity = equity
        self.cost_of_debt = cost_of_debt
        self.cost_of_equity = cost_of_equity
        self.revenue = revenue

    def calculate_eva(self) -> EVAResult:
        """Calculate EVA and related metrics."""
        # NOPAT
        nopat_value = nopat(self.operating_income, self.tax_rate)

        # Invested Capital
        ic = invested_capital(
            self.total_assets,
            self.current_liabilities,
            self.excess_cash
        )

        # WACC
        total_capital = self.debt + self.equity
        if total_capital > 0:
            debt_weight = self.debt / total_capital
            equity_weight = self.equity / total_capital
            wacc_value = (debt_weight * self.cost_of_debt * (1 - self.tax_rate) +
                          equity_weight * self.cost_of_equity)
        else:
            wacc_value = self.cost_of_equity

        # EVA
        capital_charge = ic * wacc_value
        eva_value = nopat_value - capital_charge

        # ROIC
        roic = nopat_value / ic if ic > 0 else 0
        spread = roic - wacc_value

        # EVA Margin
        margin = eva_margin(eva_value, self.revenue)

        return EVAResult(
            eva=eva_value,
            nopat=nopat_value,
            invested_capital=ic,
            wacc=wacc_value,
            capital_charge=capital_charge,
            eva_spread=spread,
            eva_margin=margin
        )

    def calculate_mva(self, market_value: float) -> float:
        """Calculate Market Value Added."""
        ic = invested_capital(
            self.total_assets,
            self.current_liabilities,
            self.excess_cash
        )
        return market_value_added(market_value, ic)

    def project_eva(
        self,
        years: int,
        nopat_growth: float,
        capital_growth: float
    ) -> List[float]:
        """
        Project future EVA.

        Args:
            years: Number of years to project
            nopat_growth: Annual NOPAT growth rate
            capital_growth: Annual invested capital growth rate

        Returns:
            List of projected EVA values
        """
        eva_result = self.calculate_eva()

        projections = []
        nopat_current = eva_result.nopat
        ic_current = eva_result.invested_capital

        for _ in range(years):
            nopat_current *= (1 + nopat_growth)
            ic_current *= (1 + capital_growth)
            eva_proj = nopat_current - (ic_current * eva_result.wacc)
            projections.append(eva_proj)

        return projections

    def value_from_eva(
        self,
        eva_projections: List[float],
        terminal_growth: float = 0.02
    ) -> float:
        """
        Calculate firm value from EVA projections.

        Firm Value = Invested Capital + MVA
        where MVA = PV of future EVA

        Args:
            eva_projections: Projected EVA values
            terminal_growth: Terminal growth rate

        Returns:
            Firm value
        """
        eva_result = self.calculate_eva()

        mva = mva_from_eva(
            eva_projections,
            eva_result.wacc,
            terminal_growth
        )

        return eva_result.invested_capital + mva


class ResidualIncomeAnalysis:
    """
    Residual Income Valuation Analysis.

    Equity valuation using the residual income approach.
    """

    def __init__(
        self,
        book_value: float,
        net_income: float,
        cost_of_equity: float,
        shares_outstanding: float = 1.0
    ):
        """Initialize RI analysis."""
        self.book_value = book_value
        self.net_income = net_income
        self.cost_of_equity = cost_of_equity
        self.shares = shares_outstanding

    @property
    def roe(self) -> Optional[float]:
        """Return on equity."""
        if self.book_value <= 0:
            return None
        return self.net_income / self.book_value

    def calculate_ri(self) -> float:
        """Calculate current residual income."""
        return residual_income(
            self.net_income,
            self.book_value,
            self.cost_of_equity
        )

    def single_stage_valuation(
        self,
        growth_rate: float,
        persistence: float = 1.0
    ) -> RIValuationResult:
        """
        Single-stage RI valuation.

        Args:
            growth_rate: Long-term growth rate
            persistence: RI persistence factor (0-1)

        Returns:
            Valuation result
        """
        ri = self.calculate_ri()

        if growth_rate < self.cost_of_equity and persistence > 0:
            terminal_ri = ri * (1 + growth_rate) * persistence
            pv_ri = terminal_ri / (self.cost_of_equity - growth_rate)
        else:
            pv_ri = ri * persistence / self.cost_of_equity

        intrinsic = self.book_value + pv_ri
        premium = (intrinsic - self.book_value) / self.book_value if self.book_value > 0 else 0

        return RIValuationResult(
            intrinsic_value=intrinsic,
            book_value=self.book_value,
            pv_residual_income=pv_ri,
            terminal_value=pv_ri,
            premium_to_book=premium
        )

    def multi_stage_valuation(
        self,
        ri_forecasts: List[float],
        terminal_growth: float,
        persistence: float = 0.5
    ) -> RIValuationResult:
        """
        Multi-stage RI valuation.

        Args:
            ri_forecasts: Explicit period RI forecasts
            terminal_growth: Terminal growth rate
            persistence: Terminal RI persistence

        Returns:
            Valuation result
        """
        intrinsic = residual_income_valuation(
            self.book_value,
            ri_forecasts,
            self.cost_of_equity,
            terminal_growth,
            persistence
        )

        # Calculate components
        pv_ri = sum(
            ri / ((1 + self.cost_of_equity) ** (t + 1))
            for t, ri in enumerate(ri_forecasts)
        )

        terminal = intrinsic - self.book_value - pv_ri
        premium = (intrinsic - self.book_value) / self.book_value if self.book_value > 0 else 0

        return RIValuationResult(
            intrinsic_value=intrinsic,
            book_value=self.book_value,
            pv_residual_income=pv_ri,
            terminal_value=terminal,
            premium_to_book=premium
        )

    def justified_multiples(self, growth_rate: float) -> Dict[str, Optional[float]]:
        """
        Calculate justified valuation multiples.

        Args:
            growth_rate: Expected growth rate

        Returns:
            Dictionary of justified multiples
        """
        roe = self.roe

        pb = justified_pb_ratio(roe, self.cost_of_equity, growth_rate) if roe else None

        # Justified P/E = P/B × (1/ROE)
        pe = pb / roe if pb and roe and roe > 0 else None

        return {
            'justified_pb': pb,
            'justified_pe': pe,
            'roe': roe,
            'cost_of_equity': self.cost_of_equity,
            'growth_rate': growth_rate,
        }
