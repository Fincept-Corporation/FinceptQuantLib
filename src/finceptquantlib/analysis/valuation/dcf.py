"""
Discounted Cash Flow (DCF) Valuation Models

Comprehensive DCF implementation:
- FCFF Valuation (Enterprise Value)
- FCFE Valuation (Equity Value)
- Dividend Discount Model (DDM)
- Terminal Value Methods
- WACC Calculation
- Sensitivity Analysis

These models estimate intrinsic value from projected cash flows.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


@dataclass
class DCFResult:
    """Result of DCF valuation."""
    enterprise_value: float = 0.0
    equity_value: float = 0.0
    value_per_share: float = 0.0
    implied_premium: float = 0.0  # vs current price

    # Components
    pv_fcf: float = 0.0
    terminal_value: float = 0.0
    pv_terminal_value: float = 0.0

    # Assumptions
    discount_rate: float = 0.0
    terminal_growth: float = 0.0


@dataclass
class WACCComponents:
    """WACC calculation breakdown."""
    cost_of_equity: float = 0.0
    cost_of_debt: float = 0.0
    weight_equity: float = 0.0
    weight_debt: float = 0.0
    tax_rate: float = 0.0
    wacc: float = 0.0


# =============================================================================
# COST OF CAPITAL CALCULATIONS
# =============================================================================

def cost_of_equity_capm(
    risk_free_rate: float,
    market_risk_premium: float,
    beta: float,
    size_premium: float = 0.0,
    country_premium: float = 0.0
) -> float:
    """
    Cost of Equity using Capital Asset Pricing Model (CAPM)

    Ke = Rf + β × (Rm - Rf) + Size Premium + Country Premium

    Args:
        risk_free_rate: Risk-free rate (e.g., 10-year Treasury)
        market_risk_premium: Expected market return minus risk-free rate
        beta: Company's beta (systematic risk)
        size_premium: Small company premium (optional)
        country_premium: Emerging market premium (optional)

    Returns: Cost of equity as decimal
    """
    return risk_free_rate + beta * market_risk_premium + size_premium + country_premium


def cost_of_equity_buildup(
    risk_free_rate: float,
    equity_risk_premium: float,
    size_premium: float = 0.0,
    specific_risk_premium: float = 0.0,
    industry_risk_premium: float = 0.0
) -> float:
    """
    Cost of Equity using Build-Up Method

    Ke = Rf + ERP + Size Premium + Specific Risk + Industry Risk

    Alternative to CAPM for private companies or when beta is unreliable.
    """
    return (
        risk_free_rate +
        equity_risk_premium +
        size_premium +
        specific_risk_premium +
        industry_risk_premium
    )


def cost_of_debt(
    interest_expense: float,
    total_debt: float,
    tax_rate: float
) -> Tuple[float, float]:
    """
    Calculate cost of debt.

    Returns both pre-tax and after-tax cost of debt.

    Args:
        interest_expense: Annual interest expense
        total_debt: Total debt
        tax_rate: Marginal tax rate

    Returns: (pre_tax_cost, after_tax_cost)
    """
    if total_debt == 0:
        return 0.0, 0.0

    pre_tax = interest_expense / total_debt
    after_tax = pre_tax * (1 - tax_rate)

    return pre_tax, after_tax


def wacc(
    cost_of_equity: float,
    cost_of_debt_pretax: float,
    market_value_equity: float,
    market_value_debt: float,
    tax_rate: float
) -> WACCComponents:
    """
    Weighted Average Cost of Capital

    WACC = (E/V) × Ke + (D/V) × Kd × (1 - T)

    Where:
    - E = Market value of equity
    - D = Market value of debt
    - V = E + D
    - Ke = Cost of equity
    - Kd = Cost of debt (pre-tax)
    - T = Tax rate

    Returns: WACCComponents with full breakdown
    """
    total_value = market_value_equity + market_value_debt

    if total_value == 0:
        return WACCComponents()

    weight_equity = market_value_equity / total_value
    weight_debt = market_value_debt / total_value

    after_tax_debt = cost_of_debt_pretax * (1 - tax_rate)

    wacc_value = weight_equity * cost_of_equity + weight_debt * after_tax_debt

    return WACCComponents(
        cost_of_equity=cost_of_equity,
        cost_of_debt=cost_of_debt_pretax,
        weight_equity=weight_equity,
        weight_debt=weight_debt,
        tax_rate=tax_rate,
        wacc=wacc_value
    )


# =============================================================================
# TERMINAL VALUE CALCULATIONS
# =============================================================================

def terminal_value_perpetuity(
    final_fcf: float,
    growth_rate: float,
    discount_rate: float
) -> float:
    """
    Terminal Value using Gordon Growth Model (Perpetuity)

    TV = FCF × (1 + g) / (r - g)

    Args:
        final_fcf: Final year free cash flow
        growth_rate: Perpetual growth rate (typically 2-3%)
        discount_rate: WACC or cost of equity

    Returns: Terminal value

    Note: growth_rate must be less than discount_rate
    """
    if growth_rate >= discount_rate:
        raise ValueError("Growth rate must be less than discount rate for perpetuity formula")

    return final_fcf * (1 + growth_rate) / (discount_rate - growth_rate)


def terminal_value_exit_multiple(
    final_metric: float,
    multiple: float
) -> float:
    """
    Terminal Value using Exit Multiple Method

    TV = Final Year Metric × Exit Multiple

    Common multiples:
    - EV/EBITDA: 6-12x
    - EV/EBIT: 8-15x
    - P/E: 12-25x

    Args:
        final_metric: Final year EBITDA, EBIT, or Net Income
        multiple: Exit multiple to apply

    Returns: Terminal value
    """
    return final_metric * multiple


def terminal_value_convergence(
    final_fcf: float,
    growth_rate: float,
    discount_rate: float,
    years_to_fade: int = 10
) -> float:
    """
    Terminal Value using Growth Rate Convergence

    More conservative approach: fade growth rate to GDP growth over time.

    Args:
        final_fcf: Final year FCF
        growth_rate: Initial terminal growth rate
        discount_rate: Discount rate
        years_to_fade: Years over which to fade growth

    Returns: Terminal value
    """
    # Linear fade from growth_rate to 2% (assumed long-term GDP)
    long_term_growth = 0.02
    fade_per_year = (growth_rate - long_term_growth) / years_to_fade

    total_tv = 0.0
    fcf = final_fcf

    for year in range(1, years_to_fade + 1):
        current_growth = growth_rate - fade_per_year * (year - 1)
        fcf = fcf * (1 + current_growth)
        pv = fcf / ((1 + discount_rate) ** year)
        total_tv += pv

    # Add perpetuity value at long-term growth rate
    perpetuity = terminal_value_perpetuity(fcf, long_term_growth, discount_rate)
    pv_perpetuity = perpetuity / ((1 + discount_rate) ** years_to_fade)
    total_tv += pv_perpetuity

    return total_tv


# =============================================================================
# DCF VALUATION MODELS
# =============================================================================

def fcff_valuation(
    fcff_projections: List[float],
    terminal_growth: float,
    wacc_rate: float,
    current_debt: float = 0.0,
    cash: float = 0.0,
    shares_outstanding: float = 1.0
) -> DCFResult:
    """
    Free Cash Flow to Firm (FCFF) Valuation

    Enterprise Value = Σ FCFF_t / (1 + WACC)^t + Terminal Value / (1 + WACC)^n

    Equity Value = Enterprise Value - Net Debt

    Args:
        fcff_projections: List of projected FCFF for each year
        terminal_growth: Perpetual growth rate
        wacc_rate: Weighted average cost of capital
        current_debt: Total debt for equity bridge
        cash: Cash and equivalents for equity bridge
        shares_outstanding: Shares for per-share value

    Returns: DCFResult with full valuation
    """
    if not fcff_projections:
        return DCFResult()

    # Discount projected FCFF
    pv_fcf = 0.0
    for i, fcf in enumerate(fcff_projections):
        year = i + 1
        pv = fcf / ((1 + wacc_rate) ** year)
        pv_fcf += pv

    # Calculate terminal value
    final_fcf = fcff_projections[-1]
    n_years = len(fcff_projections)

    if terminal_growth >= wacc_rate:
        # Use exit multiple instead if growth too high
        terminal_value = final_fcf * 10  # Default 10x FCF
    else:
        terminal_value = terminal_value_perpetuity(final_fcf, terminal_growth, wacc_rate)

    pv_terminal = terminal_value / ((1 + wacc_rate) ** n_years)

    # Enterprise value
    enterprise_value = pv_fcf + pv_terminal

    # Equity value (equity bridge)
    net_debt = current_debt - cash
    equity_value = enterprise_value - net_debt

    # Per share value
    value_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0.0

    return DCFResult(
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        value_per_share=value_per_share,
        pv_fcf=pv_fcf,
        terminal_value=terminal_value,
        pv_terminal_value=pv_terminal,
        discount_rate=wacc_rate,
        terminal_growth=terminal_growth
    )


def fcfe_valuation(
    fcfe_projections: List[float],
    terminal_growth: float,
    cost_of_equity: float,
    shares_outstanding: float = 1.0
) -> DCFResult:
    """
    Free Cash Flow to Equity (FCFE) Valuation

    Equity Value = Σ FCFE_t / (1 + Ke)^t + Terminal Value / (1 + Ke)^n

    Direct equity valuation - no equity bridge needed.

    Args:
        fcfe_projections: List of projected FCFE for each year
        terminal_growth: Perpetual growth rate
        cost_of_equity: Cost of equity (Ke)
        shares_outstanding: Shares for per-share value

    Returns: DCFResult with equity valuation
    """
    if not fcfe_projections:
        return DCFResult()

    # Discount projected FCFE
    pv_fcf = 0.0
    for i, fcf in enumerate(fcfe_projections):
        year = i + 1
        pv = fcf / ((1 + cost_of_equity) ** year)
        pv_fcf += pv

    # Calculate terminal value
    final_fcf = fcfe_projections[-1]
    n_years = len(fcfe_projections)

    if terminal_growth >= cost_of_equity:
        terminal_value = final_fcf * 15  # Default P/E equivalent
    else:
        terminal_value = terminal_value_perpetuity(final_fcf, terminal_growth, cost_of_equity)

    pv_terminal = terminal_value / ((1 + cost_of_equity) ** n_years)

    # Equity value
    equity_value = pv_fcf + pv_terminal

    # Per share value
    value_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0.0

    return DCFResult(
        enterprise_value=equity_value,  # For FCFE, EV = Equity Value
        equity_value=equity_value,
        value_per_share=value_per_share,
        pv_fcf=pv_fcf,
        terminal_value=terminal_value,
        pv_terminal_value=pv_terminal,
        discount_rate=cost_of_equity,
        terminal_growth=terminal_growth
    )


def ddm_valuation(
    dividend_projections: List[float],
    terminal_growth: float,
    cost_of_equity: float,
    shares_outstanding: float = 1.0
) -> DCFResult:
    """
    Dividend Discount Model (DDM) Valuation

    Value = Σ DPS_t / (1 + Ke)^t + Terminal Value / (1 + Ke)^n

    Best for mature dividend-paying companies.

    Args:
        dividend_projections: List of projected total dividends
        terminal_growth: Perpetual dividend growth rate
        cost_of_equity: Cost of equity
        shares_outstanding: Shares for per-share value

    Returns: DCFResult with equity valuation
    """
    return fcfe_valuation(dividend_projections, terminal_growth, cost_of_equity, shares_outstanding)


def gordon_growth_model(
    current_dividend: float,
    growth_rate: float,
    cost_of_equity: float
) -> float:
    """
    Gordon Growth Model (Constant Growth DDM)

    P0 = D0 × (1 + g) / (Ke - g) = D1 / (Ke - g)

    Simple single-stage DDM for stable dividend growers.

    Args:
        current_dividend: Current annual dividend per share
        growth_rate: Expected dividend growth rate
        cost_of_equity: Cost of equity

    Returns: Intrinsic value per share
    """
    if growth_rate >= cost_of_equity:
        raise ValueError("Growth rate must be less than cost of equity")

    d1 = current_dividend * (1 + growth_rate)
    return d1 / (cost_of_equity - growth_rate)


def two_stage_ddm(
    current_dividend: float,
    high_growth_rate: float,
    stable_growth_rate: float,
    cost_of_equity: float,
    high_growth_years: int = 5
) -> float:
    """
    Two-Stage Dividend Discount Model

    For companies with high near-term growth transitioning to stable growth.

    Args:
        current_dividend: Current dividend per share
        high_growth_rate: Growth rate for first stage
        stable_growth_rate: Terminal growth rate
        cost_of_equity: Cost of equity
        high_growth_years: Number of years in high growth phase

    Returns: Intrinsic value per share
    """
    if stable_growth_rate >= cost_of_equity:
        raise ValueError("Stable growth rate must be less than cost of equity")

    # Present value of high growth dividends
    pv_high_growth = 0.0
    dividend = current_dividend
    for year in range(1, high_growth_years + 1):
        dividend = dividend * (1 + high_growth_rate)
        pv = dividend / ((1 + cost_of_equity) ** year)
        pv_high_growth += pv

    # Terminal value at end of high growth phase
    terminal_dividend = dividend * (1 + stable_growth_rate)
    terminal_value = terminal_dividend / (cost_of_equity - stable_growth_rate)
    pv_terminal = terminal_value / ((1 + cost_of_equity) ** high_growth_years)

    return pv_high_growth + pv_terminal


# =============================================================================
# MULTI-STAGE DCF MODELS
# =============================================================================

@dataclass
class MultiStageDCFResult:
    """Result of multi-stage DCF valuation."""
    enterprise_value: float = 0.0
    equity_value: float = 0.0
    value_per_share: float = 0.0

    # Stage breakdown
    pv_stage1: float = 0.0
    pv_stage2: float = 0.0
    pv_stage3: float = 0.0
    pv_terminal: float = 0.0
    terminal_value: float = 0.0

    # Assumptions
    stage1_growth: float = 0.0
    stage2_growth: float = 0.0
    terminal_growth: float = 0.0
    discount_rate: float = 0.0


def two_stage_dcf(
    base_fcf: float,
    stage1_growth: float,
    stage1_years: int,
    terminal_growth: float,
    discount_rate: float,
    current_debt: float = 0.0,
    cash: float = 0.0,
    shares_outstanding: float = 1.0
) -> MultiStageDCFResult:
    """
    Two-Stage DCF Model

    Stage 1: High growth period (explicit forecast)
    Stage 2: Terminal value at perpetual growth

    Value = Σ FCF_t / (1+r)^t + TV / (1+r)^n

    Args:
        base_fcf: Current year free cash flow (Year 0)
        stage1_growth: Growth rate for Stage 1 (high growth)
        stage1_years: Number of years in Stage 1
        terminal_growth: Perpetual growth rate for terminal value
        discount_rate: WACC or cost of equity
        current_debt: Total debt for equity bridge
        cash: Cash for equity bridge
        shares_outstanding: Shares for per-share value

    Returns: MultiStageDCFResult with full breakdown

    Example:
        # Company with 15% growth for 5 years, then 2.5% perpetually
        result = two_stage_dcf(
            base_fcf=100,
            stage1_growth=0.15,
            stage1_years=5,
            terminal_growth=0.025,
            discount_rate=0.10
        )
    """
    if terminal_growth >= discount_rate:
        raise ValueError("Terminal growth must be less than discount rate")

    # Stage 1: High growth period
    pv_stage1 = 0.0
    fcf = base_fcf
    stage1_fcfs = []

    for year in range(1, stage1_years + 1):
        fcf = fcf * (1 + stage1_growth)
        stage1_fcfs.append(fcf)
        pv = fcf / ((1 + discount_rate) ** year)
        pv_stage1 += pv

    # Terminal Value (Gordon Growth at end of Stage 1)
    final_fcf = stage1_fcfs[-1] if stage1_fcfs else base_fcf
    terminal_fcf = final_fcf * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / ((1 + discount_rate) ** stage1_years)

    # Enterprise Value
    enterprise_value = pv_stage1 + pv_terminal

    # Equity Value
    net_debt = current_debt - cash
    equity_value = enterprise_value - net_debt

    # Per share
    value_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0.0

    return MultiStageDCFResult(
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        value_per_share=value_per_share,
        pv_stage1=pv_stage1,
        pv_stage2=0.0,
        pv_stage3=0.0,
        pv_terminal=pv_terminal,
        terminal_value=terminal_value,
        stage1_growth=stage1_growth,
        stage2_growth=0.0,
        terminal_growth=terminal_growth,
        discount_rate=discount_rate
    )


def three_stage_dcf(
    base_fcf: float,
    stage1_growth: float,
    stage1_years: int,
    stage2_growth: float,
    stage2_years: int,
    terminal_growth: float,
    discount_rate: float,
    current_debt: float = 0.0,
    cash: float = 0.0,
    shares_outstanding: float = 1.0
) -> MultiStageDCFResult:
    """
    Three-Stage DCF Model

    Stage 1: High growth period
    Stage 2: Transition/fade period
    Stage 3: Terminal value at mature growth

    Value = Σ FCF_Stage1 / (1+r)^t + Σ FCF_Stage2 / (1+r)^t + TV / (1+r)^n

    Args:
        base_fcf: Current year free cash flow (Year 0)
        stage1_growth: Growth rate for Stage 1 (high growth)
        stage1_years: Number of years in Stage 1
        stage2_growth: Growth rate for Stage 2 (transition)
        stage2_years: Number of years in Stage 2
        terminal_growth: Perpetual growth rate for terminal value
        discount_rate: WACC or cost of equity
        current_debt: Total debt for equity bridge
        cash: Cash for equity bridge
        shares_outstanding: Shares for per-share value

    Returns: MultiStageDCFResult with full breakdown

    Example:
        # High growth (20%) for 5 years, transition (10%) for 5 years, then 2.5%
        result = three_stage_dcf(
            base_fcf=100,
            stage1_growth=0.20,
            stage1_years=5,
            stage2_growth=0.10,
            stage2_years=5,
            terminal_growth=0.025,
            discount_rate=0.10
        )
    """
    if terminal_growth >= discount_rate:
        raise ValueError("Terminal growth must be less than discount rate")

    # Stage 1: High growth period
    pv_stage1 = 0.0
    fcf = base_fcf

    for year in range(1, stage1_years + 1):
        fcf = fcf * (1 + stage1_growth)
        pv = fcf / ((1 + discount_rate) ** year)
        pv_stage1 += pv

    # Stage 2: Transition period
    pv_stage2 = 0.0
    stage2_start_year = stage1_years + 1

    for year in range(stage2_start_year, stage2_start_year + stage2_years):
        fcf = fcf * (1 + stage2_growth)
        pv = fcf / ((1 + discount_rate) ** year)
        pv_stage2 += pv

    # Terminal Value at end of Stage 2
    total_explicit_years = stage1_years + stage2_years
    terminal_fcf = fcf * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / ((1 + discount_rate) ** total_explicit_years)

    # Enterprise Value
    enterprise_value = pv_stage1 + pv_stage2 + pv_terminal

    # Equity Value
    net_debt = current_debt - cash
    equity_value = enterprise_value - net_debt

    # Per share
    value_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0.0

    return MultiStageDCFResult(
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        value_per_share=value_per_share,
        pv_stage1=pv_stage1,
        pv_stage2=pv_stage2,
        pv_stage3=0.0,
        pv_terminal=pv_terminal,
        terminal_value=terminal_value,
        stage1_growth=stage1_growth,
        stage2_growth=stage2_growth,
        terminal_growth=terminal_growth,
        discount_rate=discount_rate
    )


def three_stage_dcf_fade(
    base_fcf: float,
    stage1_growth: float,
    stage1_years: int,
    stage2_years: int,
    terminal_growth: float,
    discount_rate: float,
    current_debt: float = 0.0,
    cash: float = 0.0,
    shares_outstanding: float = 1.0
) -> MultiStageDCFResult:
    """
    Three-Stage DCF with Linear Growth Fade

    Stage 1: High growth period (constant growth)
    Stage 2: Linear fade from Stage 1 growth to terminal growth
    Stage 3: Terminal value at mature growth

    More realistic than abrupt transitions.

    Args:
        base_fcf: Current year free cash flow (Year 0)
        stage1_growth: Growth rate for Stage 1 (high growth)
        stage1_years: Number of years in Stage 1
        stage2_years: Number of years to fade to terminal growth
        terminal_growth: Perpetual growth rate for terminal value
        discount_rate: WACC or cost of equity
        current_debt: Total debt for equity bridge
        cash: Cash for equity bridge
        shares_outstanding: Shares for per-share value

    Returns: MultiStageDCFResult with full breakdown

    Example:
        # 20% growth for 5 years, linear fade over 5 years to 2.5%
        result = three_stage_dcf_fade(
            base_fcf=100,
            stage1_growth=0.20,
            stage1_years=5,
            stage2_years=5,
            terminal_growth=0.025,
            discount_rate=0.10
        )
    """
    if terminal_growth >= discount_rate:
        raise ValueError("Terminal growth must be less than discount rate")

    # Stage 1: High growth period
    pv_stage1 = 0.0
    fcf = base_fcf

    for year in range(1, stage1_years + 1):
        fcf = fcf * (1 + stage1_growth)
        pv = fcf / ((1 + discount_rate) ** year)
        pv_stage1 += pv

    # Stage 2: Linear fade period
    pv_stage2 = 0.0
    growth_decline_per_year = (stage1_growth - terminal_growth) / stage2_years

    for i in range(stage2_years):
        year = stage1_years + i + 1
        # Linear interpolation of growth rate
        current_growth = stage1_growth - growth_decline_per_year * (i + 1)
        fcf = fcf * (1 + current_growth)
        pv = fcf / ((1 + discount_rate) ** year)
        pv_stage2 += pv

    # Terminal Value at end of fade period
    total_explicit_years = stage1_years + stage2_years
    terminal_fcf = fcf * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / ((1 + discount_rate) ** total_explicit_years)

    # Enterprise Value
    enterprise_value = pv_stage1 + pv_stage2 + pv_terminal

    # Equity Value
    net_debt = current_debt - cash
    equity_value = enterprise_value - net_debt

    # Per share
    value_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0.0

    # Calculate average stage 2 growth for reporting
    avg_stage2_growth = (stage1_growth + terminal_growth) / 2

    return MultiStageDCFResult(
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        value_per_share=value_per_share,
        pv_stage1=pv_stage1,
        pv_stage2=pv_stage2,
        pv_stage3=0.0,
        pv_terminal=pv_terminal,
        terminal_value=terminal_value,
        stage1_growth=stage1_growth,
        stage2_growth=avg_stage2_growth,
        terminal_growth=terminal_growth,
        discount_rate=discount_rate
    )


def h_model_dcf(
    base_fcf: float,
    initial_growth: float,
    half_life_years: float,
    terminal_growth: float,
    discount_rate: float,
    current_debt: float = 0.0,
    cash: float = 0.0,
    shares_outstanding: float = 1.0
) -> MultiStageDCFResult:
    """
    H-Model DCF Valuation

    Fuller and Hsia (1984) model for smooth growth decay.
    Growth declines linearly from initial rate to terminal rate.

    Value = FCF × (1 + g_n) / (r - g_n) + FCF × H × (g_s - g_n) / (r - g_n)

    Where:
    - g_s = initial (short-term) growth rate
    - g_n = terminal (normal) growth rate
    - H = half-life of high growth period
    - r = discount rate

    Args:
        base_fcf: Current year free cash flow
        initial_growth: Initial high growth rate
        half_life_years: Years until growth is halfway to terminal
        terminal_growth: Long-term stable growth rate
        discount_rate: WACC or cost of equity
        current_debt: Total debt for equity bridge
        cash: Cash for equity bridge
        shares_outstanding: Shares for per-share value

    Returns: MultiStageDCFResult

    Example:
        # Growth starting at 25%, declining to 3% with 5-year half-life
        result = h_model_dcf(
            base_fcf=100,
            initial_growth=0.25,
            half_life_years=5,
            terminal_growth=0.03,
            discount_rate=0.10
        )
    """
    if terminal_growth >= discount_rate:
        raise ValueError("Terminal growth must be less than discount rate")

    # H-Model formula
    # Value = FCF × (1 + g_n) / (r - g_n) + FCF × H × (g_s - g_n) / (r - g_n)

    # Terminal value component (stable growth)
    stable_value = base_fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)

    # Growth premium component
    growth_premium = base_fcf * half_life_years * (initial_growth - terminal_growth) / (discount_rate - terminal_growth)

    enterprise_value = stable_value + growth_premium

    # Equity Value
    net_debt = current_debt - cash
    equity_value = enterprise_value - net_debt

    # Per share
    value_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0.0

    return MultiStageDCFResult(
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        value_per_share=value_per_share,
        pv_stage1=growth_premium,
        pv_stage2=0.0,
        pv_stage3=0.0,
        pv_terminal=stable_value,
        terminal_value=stable_value,
        stage1_growth=initial_growth,
        stage2_growth=(initial_growth + terminal_growth) / 2,
        terminal_growth=terminal_growth,
        discount_rate=discount_rate
    )


def n_stage_dcf(
    base_fcf: float,
    growth_rates: List[float],
    years_per_stage: List[int],
    terminal_growth: float,
    discount_rate: float,
    current_debt: float = 0.0,
    cash: float = 0.0,
    shares_outstanding: float = 1.0
) -> MultiStageDCFResult:
    """
    N-Stage DCF Model (Flexible multi-stage)

    Allows any number of growth stages before terminal value.

    Args:
        base_fcf: Current year free cash flow
        growth_rates: List of growth rates for each stage
        years_per_stage: List of years for each stage (must match growth_rates length)
        terminal_growth: Perpetual growth rate
        discount_rate: WACC or cost of equity
        current_debt: Total debt for equity bridge
        cash: Cash for equity bridge
        shares_outstanding: Shares for per-share value

    Returns: MultiStageDCFResult

    Example:
        # 4 stages: 30% (3 yrs), 20% (3 yrs), 10% (4 yrs), then 2.5% perpetual
        result = n_stage_dcf(
            base_fcf=100,
            growth_rates=[0.30, 0.20, 0.10],
            years_per_stage=[3, 3, 4],
            terminal_growth=0.025,
            discount_rate=0.10
        )
    """
    if len(growth_rates) != len(years_per_stage):
        raise ValueError("growth_rates and years_per_stage must have same length")

    if terminal_growth >= discount_rate:
        raise ValueError("Terminal growth must be less than discount rate")

    fcf = base_fcf
    total_pv = 0.0
    current_year = 0
    stage_pvs = []

    # Calculate PV for each stage
    for stage_idx, (growth, years) in enumerate(zip(growth_rates, years_per_stage)):
        stage_pv = 0.0
        for _ in range(years):
            current_year += 1
            fcf = fcf * (1 + growth)
            pv = fcf / ((1 + discount_rate) ** current_year)
            stage_pv += pv
        stage_pvs.append(stage_pv)
        total_pv += stage_pv

    # Terminal Value
    terminal_fcf = fcf * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / ((1 + discount_rate) ** current_year)

    enterprise_value = total_pv + pv_terminal

    # Equity Value
    net_debt = current_debt - cash
    equity_value = enterprise_value - net_debt

    # Per share
    value_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0.0

    return MultiStageDCFResult(
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        value_per_share=value_per_share,
        pv_stage1=stage_pvs[0] if len(stage_pvs) > 0 else 0.0,
        pv_stage2=stage_pvs[1] if len(stage_pvs) > 1 else 0.0,
        pv_stage3=sum(stage_pvs[2:]) if len(stage_pvs) > 2 else 0.0,
        pv_terminal=pv_terminal,
        terminal_value=terminal_value,
        stage1_growth=growth_rates[0] if growth_rates else 0.0,
        stage2_growth=growth_rates[1] if len(growth_rates) > 1 else 0.0,
        terminal_growth=terminal_growth,
        discount_rate=discount_rate
    )


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

def dcf_sensitivity_analysis(
    base_fcf: float,
    wacc_range: Tuple[float, float],
    growth_range: Tuple[float, float],
    n_steps: int = 5
) -> Dict[str, List[List[float]]]:
    """
    Sensitivity analysis for DCF valuation.

    Creates a matrix of valuations across different WACC and growth assumptions.

    Args:
        base_fcf: Base free cash flow for valuation
        wacc_range: (min_wacc, max_wacc) tuple
        growth_range: (min_growth, max_growth) tuple
        n_steps: Number of steps in each dimension

    Returns: Dictionary with WACC values, growth values, and valuation matrix
    """
    wacc_min, wacc_max = wacc_range
    growth_min, growth_max = growth_range

    wacc_step = (wacc_max - wacc_min) / (n_steps - 1)
    growth_step = (growth_max - growth_min) / (n_steps - 1)

    wacc_values = [wacc_min + i * wacc_step for i in range(n_steps)]
    growth_values = [growth_min + i * growth_step for i in range(n_steps)]

    # Create valuation matrix
    matrix = []
    for wacc_val in wacc_values:
        row = []
        for growth_val in growth_values:
            if growth_val >= wacc_val:
                value = float('inf')
            else:
                value = base_fcf * (1 + growth_val) / (wacc_val - growth_val)
            row.append(value)
        matrix.append(row)

    return {
        'wacc_values': wacc_values,
        'growth_values': growth_values,
        'valuation_matrix': matrix
    }


def monte_carlo_dcf(
    base_fcf: float,
    wacc_mean: float,
    wacc_std: float,
    growth_mean: float,
    growth_std: float,
    n_simulations: int = 10000,
    seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Monte Carlo simulation for DCF valuation.

    Generates distribution of values given uncertainty in inputs.

    Args:
        base_fcf: Base free cash flow
        wacc_mean: Mean WACC
        wacc_std: Standard deviation of WACC
        growth_mean: Mean terminal growth
        growth_std: Standard deviation of growth
        n_simulations: Number of simulations
        seed: Random seed for reproducibility

    Returns: Dictionary with statistics of valuation distribution
    """
    import random

    rng = random.Random(seed)

    values = []

    for _ in range(n_simulations):
        # Sample from normal distributions
        wacc_sample = rng.gauss(wacc_mean, wacc_std)
        growth_sample = rng.gauss(growth_mean, growth_std)

        # Ensure wacc > growth
        wacc_sample = max(wacc_sample, growth_sample + 0.01)

        # Calculate value
        value = base_fcf * (1 + growth_sample) / (wacc_sample - growth_sample)
        values.append(value)

    # Calculate statistics
    values.sort()
    n = len(values)

    mean_value = sum(values) / n
    variance = sum((v - mean_value) ** 2 for v in values) / n
    std_value = variance ** 0.5

    return {
        'mean': mean_value,
        'median': values[n // 2],
        'std': std_value,
        'p10': values[int(n * 0.10)],
        'p25': values[int(n * 0.25)],
        'p75': values[int(n * 0.75)],
        'p90': values[int(n * 0.90)],
        'min': values[0],
        'max': values[-1],
    }


# =============================================================================
# DCF MODEL CLASS
# =============================================================================

class DCFModel:
    """
    Comprehensive DCF valuation model.

    Example:
        # Setup assumptions
        projections = [100, 110, 121, 133, 146]  # 5-year FCFF projections
        model = DCFModel(
            fcf_projections=projections,
            wacc=0.10,
            terminal_growth=0.025,
            current_debt=200,
            cash=50,
            shares=100
        )

        # Run valuation
        result = model.fcff_value()
        print(f"Value per share: ${result.value_per_share:.2f}")

        # Sensitivity analysis
        sensitivity = model.sensitivity()
    """

    def __init__(
        self,
        fcf_projections: List[float],
        wacc: float,
        terminal_growth: float,
        current_debt: float = 0.0,
        cash: float = 0.0,
        shares: float = 1.0,
        cost_of_equity: Optional[float] = None
    ):
        self.fcf_projections = fcf_projections
        self.wacc = wacc
        self.terminal_growth = terminal_growth
        self.current_debt = current_debt
        self.cash = cash
        self.shares = shares
        self.cost_of_equity = cost_of_equity or wacc + 0.03  # Approximate if not provided

    def fcff_value(self) -> DCFResult:
        """Calculate FCFF-based enterprise value."""
        return fcff_valuation(
            self.fcf_projections,
            self.terminal_growth,
            self.wacc,
            self.current_debt,
            self.cash,
            self.shares
        )

    def fcfe_value(self, fcfe_projections: Optional[List[float]] = None) -> DCFResult:
        """Calculate FCFE-based equity value."""
        fcfe = fcfe_projections or self.fcf_projections
        return fcfe_valuation(
            fcfe,
            self.terminal_growth,
            self.cost_of_equity,
            self.shares
        )

    def implied_premium(self, current_price: float) -> float:
        """Calculate implied premium/discount to current price."""
        result = self.fcff_value()
        if current_price == 0:
            return 0.0
        return (result.value_per_share - current_price) / current_price

    def sensitivity(
        self,
        wacc_range: Tuple[float, float] = None,
        growth_range: Tuple[float, float] = None
    ) -> Dict[str, List[List[float]]]:
        """Run sensitivity analysis."""
        if wacc_range is None:
            wacc_range = (self.wacc - 0.02, self.wacc + 0.02)
        if growth_range is None:
            growth_range = (self.terminal_growth - 0.01, self.terminal_growth + 0.01)

        base_fcf = self.fcf_projections[-1]
        return dcf_sensitivity_analysis(base_fcf, wacc_range, growth_range)

    def to_dict(self) -> Dict[str, float]:
        """Return valuation summary as dictionary."""
        result = self.fcff_value()
        return {
            'enterprise_value': result.enterprise_value,
            'equity_value': result.equity_value,
            'value_per_share': result.value_per_share,
            'pv_fcf': result.pv_fcf,
            'terminal_value': result.terminal_value,
            'pv_terminal': result.pv_terminal_value,
            'wacc': self.wacc,
            'terminal_growth': self.terminal_growth,
        }
