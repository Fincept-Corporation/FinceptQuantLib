"""
Predictive Models for Financial Distress and Quality

Comprehensive predictive model implementations:
- Altman Z-Score (Original, Z', Z'')
- Piotroski F-Score
- Beneish M-Score
- Springate Score
- Ohlson O-Score
- Zmijewski Score

These models predict bankruptcy, financial distress, or earnings manipulation.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from ..fundamentals.statements import (
    IncomeStatement, BalanceSheet, FinancialStatements
)
from ..fundamentals.valuation import MarketData


@dataclass
class AltmanZResult:
    """Result of Altman Z-Score analysis."""
    z_score: float = 0.0
    zone: str = ""  # "Safe", "Grey", "Distress"
    components: Dict[str, float] = None

    def __post_init__(self):
        if self.components is None:
            self.components = {}


@dataclass
class PiotroskiFResult:
    """Result of Piotroski F-Score analysis."""
    f_score: int = 0
    components: Dict[str, bool] = None
    category: str = ""  # "Strong", "Average", "Weak"

    def __post_init__(self):
        if self.components is None:
            self.components = {}


@dataclass
class BeneishMResult:
    """Result of Beneish M-Score analysis."""
    m_score: float = 0.0
    manipulation_likely: bool = False
    components: Dict[str, float] = None

    def __post_init__(self):
        if self.components is None:
            self.components = {}


# =============================================================================
# ALTMAN Z-SCORE MODELS
# =============================================================================

def altman_z_score(
    income: IncomeStatement,
    balance: BalanceSheet,
    market: MarketData
) -> AltmanZResult:
    """
    Original Altman Z-Score (1968) for Public Manufacturing Companies

    Z = 1.2×X1 + 1.4×X2 + 3.3×X3 + 0.6×X4 + 1.0×X5

    Where:
    - X1 = Working Capital / Total Assets
    - X2 = Retained Earnings / Total Assets
    - X3 = EBIT / Total Assets
    - X4 = Market Value of Equity / Book Value of Total Liabilities
    - X5 = Sales / Total Assets

    Interpretation:
    - Z > 2.99: Safe Zone
    - 1.81 ≤ Z ≤ 2.99: Grey Zone
    - Z < 1.81: Distress Zone

    Note: This model is designed for public manufacturing companies.
    """
    if balance.total_assets == 0:
        return AltmanZResult(z_score=0.0, zone="Unknown", components={})

    # X1: Working Capital / Total Assets
    x1 = balance.working_capital / balance.total_assets

    # X2: Retained Earnings / Total Assets
    x2 = balance.retained_earnings / balance.total_assets

    # X3: EBIT / Total Assets
    x3 = income.operating_income / balance.total_assets

    # X4: Market Value of Equity / Total Liabilities
    if balance.total_liabilities > 0:
        x4 = market.market_cap / balance.total_liabilities
    else:
        x4 = 0.0

    # X5: Sales / Total Assets
    x5 = income.revenue / balance.total_assets

    # Calculate Z-Score
    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

    # Determine zone
    if z > 2.99:
        zone = "Safe"
    elif z >= 1.81:
        zone = "Grey"
    else:
        zone = "Distress"

    return AltmanZResult(
        z_score=z,
        zone=zone,
        components={
            'x1_working_capital': x1,
            'x2_retained_earnings': x2,
            'x3_ebit_to_assets': x3,
            'x4_market_to_book_liab': x4,
            'x5_asset_turnover': x5,
        }
    )


def altman_z_prime(
    income: IncomeStatement,
    balance: BalanceSheet
) -> AltmanZResult:
    """
    Altman Z'-Score for Private Manufacturing Companies

    Z' = 0.717×X1 + 0.847×X2 + 3.107×X3 + 0.420×X4 + 0.998×X5

    Replaces market value with book value of equity for X4.

    Interpretation:
    - Z' > 2.90: Safe Zone
    - 1.23 ≤ Z' ≤ 2.90: Grey Zone
    - Z' < 1.23: Distress Zone
    """
    if balance.total_assets == 0:
        return AltmanZResult(z_score=0.0, zone="Unknown", components={})

    # X1: Working Capital / Total Assets
    x1 = balance.working_capital / balance.total_assets

    # X2: Retained Earnings / Total Assets
    x2 = balance.retained_earnings / balance.total_assets

    # X3: EBIT / Total Assets
    x3 = income.operating_income / balance.total_assets

    # X4: Book Value of Equity / Total Liabilities (changed from market value)
    if balance.total_liabilities > 0:
        x4 = balance.total_equity / balance.total_liabilities
    else:
        x4 = 0.0

    # X5: Sales / Total Assets
    x5 = income.revenue / balance.total_assets

    # Calculate Z'-Score
    z = 0.717 * x1 + 0.847 * x2 + 3.107 * x3 + 0.420 * x4 + 0.998 * x5

    # Determine zone
    if z > 2.90:
        zone = "Safe"
    elif z >= 1.23:
        zone = "Grey"
    else:
        zone = "Distress"

    return AltmanZResult(
        z_score=z,
        zone=zone,
        components={
            'x1_working_capital': x1,
            'x2_retained_earnings': x2,
            'x3_ebit_to_assets': x3,
            'x4_book_equity_to_liab': x4,
            'x5_asset_turnover': x5,
        }
    )


def altman_z_double_prime(
    income: IncomeStatement,
    balance: BalanceSheet
) -> AltmanZResult:
    """
    Altman Z''-Score for Non-Manufacturing and Emerging Markets

    Z'' = 6.56×X1 + 3.26×X2 + 6.72×X3 + 1.05×X4

    Removes asset turnover (X5) for non-manufacturing companies.

    Interpretation:
    - Z'' > 2.60: Safe Zone
    - 1.10 ≤ Z'' ≤ 2.60: Grey Zone
    - Z'' < 1.10: Distress Zone
    """
    if balance.total_assets == 0:
        return AltmanZResult(z_score=0.0, zone="Unknown", components={})

    # X1: Working Capital / Total Assets
    x1 = balance.working_capital / balance.total_assets

    # X2: Retained Earnings / Total Assets
    x2 = balance.retained_earnings / balance.total_assets

    # X3: EBIT / Total Assets
    x3 = income.operating_income / balance.total_assets

    # X4: Book Value of Equity / Total Liabilities
    if balance.total_liabilities > 0:
        x4 = balance.total_equity / balance.total_liabilities
    else:
        x4 = 0.0

    # Calculate Z''-Score (no X5)
    z = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4

    # Determine zone
    if z > 2.60:
        zone = "Safe"
    elif z >= 1.10:
        zone = "Grey"
    else:
        zone = "Distress"

    return AltmanZResult(
        z_score=z,
        zone=zone,
        components={
            'x1_working_capital': x1,
            'x2_retained_earnings': x2,
            'x3_ebit_to_assets': x3,
            'x4_book_equity_to_liab': x4,
        }
    )


# =============================================================================
# PIOTROSKI F-SCORE
# =============================================================================

def piotroski_f_score(
    current: FinancialStatements,
    prior: FinancialStatements
) -> PiotroskiFResult:
    """
    Piotroski F-Score (2000)

    A 9-point scoring system to identify financially strong companies.
    Originally designed to identify value stocks likely to outperform.

    Components (1 point each if true):

    Profitability:
    1. ROA > 0 (positive net income)
    2. CFO > 0 (positive operating cash flow)
    3. ROA improvement (current > prior)
    4. Accruals (CFO > Net Income)

    Leverage/Liquidity:
    5. Leverage decrease (LT Debt / Assets decreased)
    6. Current Ratio improvement
    7. No share dilution (shares not increased)

    Operating Efficiency:
    8. Gross Margin improvement
    9. Asset Turnover improvement

    Interpretation:
    - 8-9: Strong (buy signal)
    - 5-7: Average
    - 0-4: Weak (avoid/sell signal)
    """
    income = current.income_statement
    balance = current.balance_sheet
    cash_flow = current.cash_flow
    prior_income = prior.income_statement
    prior_balance = prior.balance_sheet

    components = {}

    # 1. ROA > 0
    roa = income.net_income / balance.total_assets if balance.total_assets > 0 else 0
    components['positive_roa'] = roa > 0

    # 2. CFO > 0
    components['positive_cfo'] = cash_flow.cash_from_operations > 0

    # 3. ROA Improvement
    prior_roa = prior_income.net_income / prior_balance.total_assets if prior_balance.total_assets > 0 else 0
    components['roa_improvement'] = roa > prior_roa

    # 4. Accruals (CFO > Net Income = lower accruals)
    components['cfo_exceeds_income'] = cash_flow.cash_from_operations > income.net_income

    # 5. Leverage Decrease
    current_leverage = balance.long_term_debt / balance.total_assets if balance.total_assets > 0 else 0
    prior_leverage = prior_balance.long_term_debt / prior_balance.total_assets if prior_balance.total_assets > 0 else 0
    components['leverage_decrease'] = current_leverage < prior_leverage

    # 6. Current Ratio Improvement
    current_ratio = balance.total_current_assets / balance.total_current_liabilities if balance.total_current_liabilities > 0 else 0
    prior_current_ratio = prior_balance.total_current_assets / prior_balance.total_current_liabilities if prior_balance.total_current_liabilities > 0 else 0
    components['current_ratio_improvement'] = current_ratio > prior_current_ratio

    # 7. No Share Dilution
    components['no_dilution'] = income.shares_outstanding <= prior_income.shares_outstanding

    # 8. Gross Margin Improvement
    current_gm = income.gross_profit / income.revenue if income.revenue > 0 else 0
    prior_gm = prior_income.gross_profit / prior_income.revenue if prior_income.revenue > 0 else 0
    components['gross_margin_improvement'] = current_gm > prior_gm

    # 9. Asset Turnover Improvement
    current_at = income.revenue / balance.total_assets if balance.total_assets > 0 else 0
    prior_at = prior_income.revenue / prior_balance.total_assets if prior_balance.total_assets > 0 else 0
    components['asset_turnover_improvement'] = current_at > prior_at

    # Calculate F-Score
    f_score = sum(1 for v in components.values() if v)

    # Determine category
    if f_score >= 8:
        category = "Strong"
    elif f_score >= 5:
        category = "Average"
    else:
        category = "Weak"

    return PiotroskiFResult(
        f_score=f_score,
        components=components,
        category=category
    )


# =============================================================================
# BENEISH M-SCORE
# =============================================================================

def beneish_m_score(
    current: FinancialStatements,
    prior: FinancialStatements
) -> BeneishMResult:
    """
    Beneish M-Score (1999)

    A model to detect earnings manipulation.

    M = -4.84 + 0.92×DSRI + 0.528×GMI + 0.404×AQI + 0.892×SGI
        + 0.115×DEPI - 0.172×SGAI + 4.679×TATA - 0.327×LVGI

    Components:
    - DSRI: Days Sales in Receivables Index
    - GMI: Gross Margin Index
    - AQI: Asset Quality Index
    - SGI: Sales Growth Index
    - DEPI: Depreciation Index
    - SGAI: SG&A Index
    - TATA: Total Accruals to Total Assets
    - LVGI: Leverage Index

    Interpretation:
    - M > -2.22: Likely manipulator
    - M ≤ -2.22: Unlikely manipulator
    """
    income = current.income_statement
    balance = current.balance_sheet
    cash_flow = current.cash_flow
    prior_income = prior.income_statement
    prior_balance = prior.balance_sheet

    # Calculate indices
    components = {}

    # DSRI: Days Sales in Receivables Index
    # = (AR_t / Revenue_t) / (AR_t-1 / Revenue_t-1)
    dsr_current = balance.accounts_receivable / income.revenue if income.revenue > 0 else 0
    dsr_prior = prior_balance.accounts_receivable / prior_income.revenue if prior_income.revenue > 0 else 0
    dsri = dsr_current / dsr_prior if dsr_prior > 0 else 1.0
    components['dsri'] = dsri

    # GMI: Gross Margin Index
    # = GM_t-1 / GM_t (inverted - higher means declining GM)
    gm_current = income.gross_profit / income.revenue if income.revenue > 0 else 0
    gm_prior = prior_income.gross_profit / prior_income.revenue if prior_income.revenue > 0 else 0
    gmi = gm_prior / gm_current if gm_current > 0 else 1.0
    components['gmi'] = gmi

    # AQI: Asset Quality Index
    # = [1 - (CA + PPE) / TA]_t / [1 - (CA + PPE) / TA]_t-1
    hard_assets_current = (balance.total_current_assets + balance.net_ppe) / balance.total_assets if balance.total_assets > 0 else 1
    hard_assets_prior = (prior_balance.total_current_assets + prior_balance.net_ppe) / prior_balance.total_assets if prior_balance.total_assets > 0 else 1
    aqi_current = 1 - hard_assets_current
    aqi_prior = 1 - hard_assets_prior
    aqi = aqi_current / aqi_prior if aqi_prior > 0 else 1.0
    components['aqi'] = aqi

    # SGI: Sales Growth Index
    # = Revenue_t / Revenue_t-1
    sgi = income.revenue / prior_income.revenue if prior_income.revenue > 0 else 1.0
    components['sgi'] = sgi

    # DEPI: Depreciation Index
    # = (Depr / (Depr + PPE))_t-1 / (Depr / (Depr + PPE))_t
    depr_rate_current = income.depreciation / (income.depreciation + balance.net_ppe) if (income.depreciation + balance.net_ppe) > 0 else 0
    depr_rate_prior = prior_income.depreciation / (prior_income.depreciation + prior_balance.net_ppe) if (prior_income.depreciation + prior_balance.net_ppe) > 0 else 0
    depi = depr_rate_prior / depr_rate_current if depr_rate_current > 0 else 1.0
    components['depi'] = depi

    # SGAI: SG&A Index
    # = (SGA / Revenue)_t / (SGA / Revenue)_t-1
    sga_ratio_current = income.sga_expenses / income.revenue if income.revenue > 0 else 0
    sga_ratio_prior = prior_income.sga_expenses / prior_income.revenue if prior_income.revenue > 0 else 0
    sgai = sga_ratio_current / sga_ratio_prior if sga_ratio_prior > 0 else 1.0
    components['sgai'] = sgai

    # TATA: Total Accruals to Total Assets
    # = (Net Income - CFO) / Total Assets
    tata = (income.net_income - cash_flow.cash_from_operations) / balance.total_assets if balance.total_assets > 0 else 0
    components['tata'] = tata

    # LVGI: Leverage Index
    # = [(CL + LTD) / TA]_t / [(CL + LTD) / TA]_t-1
    leverage_current = (balance.total_current_liabilities + balance.long_term_debt) / balance.total_assets if balance.total_assets > 0 else 0
    leverage_prior = (prior_balance.total_current_liabilities + prior_balance.long_term_debt) / prior_balance.total_assets if prior_balance.total_assets > 0 else 0
    lvgi = leverage_current / leverage_prior if leverage_prior > 0 else 1.0
    components['lvgi'] = lvgi

    # Calculate M-Score
    m_score = (
        -4.84
        + 0.920 * dsri
        + 0.528 * gmi
        + 0.404 * aqi
        + 0.892 * sgi
        + 0.115 * depi
        - 0.172 * sgai
        + 4.679 * tata
        - 0.327 * lvgi
    )

    # Interpretation
    manipulation_likely = m_score > -2.22

    return BeneishMResult(
        m_score=m_score,
        manipulation_likely=manipulation_likely,
        components=components
    )


# =============================================================================
# OTHER PREDICTIVE MODELS
# =============================================================================

def springate_score(
    income: IncomeStatement,
    balance: BalanceSheet
) -> Tuple[float, str]:
    """
    Springate Score (1978)

    S = 1.03×A + 3.07×B + 0.66×C + 0.4×D

    Where:
    - A = Working Capital / Total Assets
    - B = EBIT / Total Assets
    - C = EBT / Current Liabilities
    - D = Revenue / Total Assets

    Interpretation:
    - S < 0.862: Distress
    - S ≥ 0.862: Safe

    Returns: (score, zone)
    """
    if balance.total_assets == 0:
        return 0.0, "Unknown"

    a = balance.working_capital / balance.total_assets
    b = income.operating_income / balance.total_assets
    c = income.pretax_income / balance.total_current_liabilities if balance.total_current_liabilities > 0 else 0
    d = income.revenue / balance.total_assets

    s = 1.03 * a + 3.07 * b + 0.66 * c + 0.4 * d

    zone = "Safe" if s >= 0.862 else "Distress"

    return s, zone


def ohlson_o_score(
    income: IncomeStatement,
    balance: BalanceSheet
) -> Tuple[float, float, str]:
    """
    Ohlson O-Score (1980)

    Logistic regression model for bankruptcy prediction.

    O = -1.32 - 0.407×ln(TA/GNP) + 6.03×(TL/TA) - 1.43×(WC/TA)
        + 0.0757×(CL/CA) - 1.72×X - 2.37×(NI/TA) - 1.83×(FFO/TL)
        + 0.285×Y - 0.521×Z

    Where:
    - X = 1 if TL > TA, else 0
    - Y = 1 if NI<0 for last two years, else 0 (simplified: NI<0)
    - Z = (NI_t - NI_t-1) / (|NI_t| + |NI_t-1|)

    P(bankruptcy) = 1 / (1 + e^(-O))

    Returns: (o_score, probability, zone)
    """
    # Simplified implementation (some variables require historical data)
    import math

    if balance.total_assets == 0:
        return 0.0, 0.0, "Unknown"

    # Using normalized total assets (assuming GNP scaling factor of 1 for simplicity)
    size = math.log(balance.total_assets) if balance.total_assets > 0 else 0
    tlta = balance.total_liabilities / balance.total_assets
    wcta = balance.working_capital / balance.total_assets
    clca = balance.total_current_liabilities / balance.total_current_assets if balance.total_current_assets > 0 else 0
    x = 1 if balance.total_liabilities > balance.total_assets else 0
    nita = income.net_income / balance.total_assets

    # FFO approximation (Funds From Operations)
    ffo_tl = (income.net_income + income.depreciation) / balance.total_liabilities if balance.total_liabilities > 0 else 0

    # Y simplified (current year only)
    y = 1 if income.net_income < 0 else 0

    # Z simplified (no prior year, set to 0)
    z = 0

    o_score = (
        -1.32
        - 0.407 * size
        + 6.03 * tlta
        - 1.43 * wcta
        + 0.0757 * clca
        - 1.72 * x
        - 2.37 * nita
        - 1.83 * ffo_tl
        + 0.285 * y
        - 0.521 * z
    )

    probability = 1 / (1 + math.exp(-o_score))

    if probability > 0.5:
        zone = "High Risk"
    elif probability > 0.25:
        zone = "Moderate Risk"
    else:
        zone = "Low Risk"

    return o_score, probability, zone


def zmijewski_score(
    income: IncomeStatement,
    balance: BalanceSheet
) -> Tuple[float, float, str]:
    """
    Zmijewski Score (1984)

    Probit model for bankruptcy prediction.

    X = -4.3 - 4.5×(NI/TA) + 5.7×(TL/TA) - 0.004×(CA/CL)

    P(bankruptcy) = Φ(X) where Φ is standard normal CDF

    Returns: (score, probability, zone)
    """
    import math

    if balance.total_assets == 0:
        return 0.0, 0.0, "Unknown"

    nita = income.net_income / balance.total_assets
    tlta = balance.total_liabilities / balance.total_assets
    cacl = balance.total_current_assets / balance.total_current_liabilities if balance.total_current_liabilities > 0 else 0

    x = -4.3 - 4.5 * nita + 5.7 * tlta - 0.004 * cacl

    # Standard normal CDF approximation
    def norm_cdf(x):
        """Approximate standard normal CDF"""
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        p = 0.3275911
        sign = 1 if x >= 0 else -1
        x = abs(x) / math.sqrt(2)
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
        return 0.5 * (1.0 + sign * y)

    probability = norm_cdf(x)

    if probability > 0.5:
        zone = "High Risk"
    elif probability > 0.25:
        zone = "Moderate Risk"
    else:
        zone = "Low Risk"

    return x, probability, zone


# =============================================================================
# COMPREHENSIVE PREDICTIVE ANALYSIS
# =============================================================================

def comprehensive_distress_analysis(
    current: FinancialStatements,
    prior: FinancialStatements,
    market: Optional[MarketData] = None
) -> Dict[str, any]:
    """
    Run all predictive models and provide comprehensive assessment.

    Args:
        current: Current period financial statements
        prior: Prior period financial statements
        market: Market data (optional, for Altman Z original)

    Returns: Dictionary with all model results and overall assessment
    """
    income = current.income_statement
    balance = current.balance_sheet

    results = {}

    # Altman Z-Scores
    if market:
        results['altman_z'] = altman_z_score(income, balance, market)
    results['altman_z_prime'] = altman_z_prime(income, balance)
    results['altman_z_double_prime'] = altman_z_double_prime(income, balance)

    # Piotroski F-Score
    results['piotroski_f'] = piotroski_f_score(current, prior)

    # Beneish M-Score
    results['beneish_m'] = beneish_m_score(current, prior)

    # Other scores
    spring_score, spring_zone = springate_score(income, balance)
    results['springate'] = {'score': spring_score, 'zone': spring_zone}

    ohlson_score, ohlson_prob, ohlson_zone = ohlson_o_score(income, balance)
    results['ohlson'] = {'score': ohlson_score, 'probability': ohlson_prob, 'zone': ohlson_zone}

    zmij_score, zmij_prob, zmij_zone = zmijewski_score(income, balance)
    results['zmijewski'] = {'score': zmij_score, 'probability': zmij_prob, 'zone': zmij_zone}

    # Overall Assessment
    distress_signals = 0
    total_signals = 0

    # Count distress signals
    if 'altman_z' in results and results['altman_z'].zone == "Distress":
        distress_signals += 1
    total_signals += 1

    if results['altman_z_prime'].zone == "Distress":
        distress_signals += 1
    total_signals += 1

    if results['piotroski_f'].f_score <= 3:
        distress_signals += 1
    total_signals += 1

    if results['beneish_m'].manipulation_likely:
        distress_signals += 1
    total_signals += 1

    if results['springate']['zone'] == "Distress":
        distress_signals += 1
    total_signals += 1

    # Overall rating
    if distress_signals == 0:
        overall = "LOW RISK"
    elif distress_signals <= 2:
        overall = "MODERATE RISK"
    else:
        overall = "HIGH RISK"

    results['overall_assessment'] = {
        'distress_signals': distress_signals,
        'total_models': total_signals,
        'rating': overall
    }

    return results
