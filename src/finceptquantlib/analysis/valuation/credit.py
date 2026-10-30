"""
Credit Analysis Module

Structural credit models and default probability estimation:
- Merton Structural Model
- Distance to Default (DD)
- Probability of Default (PD)
- Expected Loss and Loss Given Default
- Credit Spreads from Fundamentals
- KMV-Style EDF Calculation
- Credit Scoring Models
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import math

from finceptquantlib.core.utilities.distributions import norm_cdf, norm_ppf


# =============================================================================
# Merton Structural Model
# =============================================================================

@dataclass
class MertonModelResult:
    """Results from Merton structural model."""
    asset_value: float
    asset_volatility: float
    distance_to_default: float
    probability_of_default: float
    expected_loss: float
    credit_spread: float
    d1: float
    d2: float
    equity_value: float
    debt_value: float


def merton_model(
    equity_value: float,
    equity_volatility: float,
    debt_face_value: float,
    risk_free_rate: float,
    time_to_maturity: float,
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> MertonModelResult:
    """
    Merton Structural Credit Model.

    Models the firm as a call option on its assets. Default occurs if
    asset value falls below debt face value at maturity.

    Uses iterative solving because:
    E = V*N(d1) - D*e^(-rT)*N(d2)  (equity is call on assets)
    σ_E = (V/E)*N(d1)*σ_V          (equity volatility relation)

    Args:
        equity_value: Market value of equity
        equity_volatility: Equity volatility (annualized)
        debt_face_value: Face value of debt (default point)
        risk_free_rate: Risk-free rate
        time_to_maturity: Time to maturity in years
        max_iterations: Maximum solver iterations
        tolerance: Convergence tolerance

    Returns:
        MertonModelResult with all model outputs
    """
    T = time_to_maturity
    r = risk_free_rate
    E = equity_value
    D = debt_face_value
    sigma_E = equity_volatility

    # Initial guess: Asset value = Equity + Debt, Asset vol = Equity vol
    V = E + D
    sigma_V = sigma_E * E / V

    for _ in range(max_iterations):
        # Calculate d1 and d2
        d1 = (math.log(V / D) + (r + 0.5 * sigma_V ** 2) * T) / (sigma_V * math.sqrt(T))
        d2 = d1 - sigma_V * math.sqrt(T)

        # Equity value from Black-Scholes
        E_calc = V * norm_cdf(d1) - D * math.exp(-r * T) * norm_cdf(d2)

        # Equity volatility relation
        sigma_E_calc = (V / E_calc) * norm_cdf(d1) * sigma_V if E_calc > 0 else sigma_V

        # Check convergence
        if abs(E_calc - E) < tolerance and abs(sigma_E_calc - sigma_E) < tolerance:
            break

        # Update estimates using Newton-Raphson style adjustment
        V = V * (E / E_calc) ** 0.5
        sigma_V = sigma_V * (sigma_E / sigma_E_calc) ** 0.5

        # Bounds
        V = max(V, E)
        sigma_V = max(0.01, min(sigma_V, 2.0))

    # Final calculations
    d1 = (math.log(V / D) + (r + 0.5 * sigma_V ** 2) * T) / (sigma_V * math.sqrt(T))
    d2 = d1 - sigma_V * math.sqrt(T)

    # Distance to default (in standard deviations)
    distance_to_default = d2

    # Probability of default = N(-d2)
    prob_default = norm_cdf(-d2)

    # Risk-neutral expected loss
    # Expected loss = D*e^(-rT)*N(-d2) - V*N(-d1)
    risky_debt = D * math.exp(-r * T) * norm_cdf(d2) + V * norm_cdf(-d1)
    risk_free_debt = D * math.exp(-r * T)
    expected_loss_pct = 1 - (risky_debt / risk_free_debt) if risk_free_debt > 0 else 0

    # Credit spread
    if risky_debt > 0 and T > 0:
        yield_risky = -math.log(risky_debt / D) / T
        credit_spread = yield_risky - r
    else:
        credit_spread = 0

    return MertonModelResult(
        asset_value=V,
        asset_volatility=sigma_V,
        distance_to_default=distance_to_default,
        probability_of_default=prob_default,
        expected_loss=expected_loss_pct,
        credit_spread=credit_spread,
        d1=d1,
        d2=d2,
        equity_value=E,
        debt_value=risky_debt
    )


def distance_to_default(
    asset_value: float,
    asset_volatility: float,
    default_point: float,
    time_horizon: float = 1.0,
    drift: float = 0.0
) -> float:
    """
    Distance to Default (DD).

    Number of standard deviations asset value is from default.

    DD = [ln(V/D) + (μ - σ²/2)T] / (σ√T)

    Higher DD = Lower default risk

    Typical interpretations:
    - DD > 4: Very low risk
    - DD 2-4: Low risk
    - DD 1-2: Moderate risk
    - DD < 1: High risk

    Args:
        asset_value: Current asset value
        asset_volatility: Asset volatility
        default_point: Default threshold (typically debt)
        time_horizon: Time horizon in years
        drift: Expected asset growth rate

    Returns:
        Distance to default in standard deviations
    """
    if default_point <= 0 or asset_value <= 0 or asset_volatility <= 0:
        return 0.0

    T = time_horizon
    numerator = math.log(asset_value / default_point) + (drift - 0.5 * asset_volatility ** 2) * T
    denominator = asset_volatility * math.sqrt(T)

    return numerator / denominator


def probability_of_default_from_dd(
    distance_to_default: float,
    use_empirical: bool = False
) -> float:
    """
    Probability of Default from Distance to Default.

    Risk-neutral PD: N(-DD)
    Empirical PD uses historical mapping (KMV-style)

    Args:
        distance_to_default: DD value
        use_empirical: Use empirical mapping vs theoretical

    Returns:
        Probability of default
    """
    if use_empirical:
        # Simplified empirical mapping (KMV-style)
        # In practice, this would be calibrated to historical defaults
        # Using a steeper mapping than theoretical normal
        if distance_to_default >= 6:
            return 0.0001
        elif distance_to_default >= 4:
            return 0.001
        elif distance_to_default >= 3:
            return 0.005
        elif distance_to_default >= 2:
            return 0.02
        elif distance_to_default >= 1:
            return 0.08
        elif distance_to_default >= 0:
            return 0.20
        else:
            return min(0.5 + 0.2 * abs(distance_to_default), 1.0)
    else:
        # Theoretical (risk-neutral)
        return norm_cdf(-distance_to_default)


def expected_default_frequency(
    equity_value: float,
    equity_volatility: float,
    short_term_debt: float,
    long_term_debt: float,
    time_horizon: float = 1.0
) -> Dict[str, float]:
    """
    KMV-Style Expected Default Frequency (EDF).

    Simplified KMV approach:
    1. Default Point = STD + 0.5 * LTD
    2. Calculate DD
    3. Map DD to EDF using empirical distribution

    Args:
        equity_value: Market cap
        equity_volatility: Equity volatility
        short_term_debt: Short-term debt
        long_term_debt: Long-term debt
        time_horizon: Time horizon (typically 1 year)

    Returns:
        Dictionary with DD and EDF
    """
    # Default point (KMV convention)
    default_point = short_term_debt + 0.5 * long_term_debt

    # Approximate asset value and volatility
    total_debt = short_term_debt + long_term_debt
    asset_value = equity_value + total_debt
    asset_volatility = equity_volatility * (equity_value / asset_value)

    # Distance to default
    dd = distance_to_default(
        asset_value,
        asset_volatility,
        default_point,
        time_horizon
    )

    # EDF from empirical mapping
    edf = probability_of_default_from_dd(dd, use_empirical=True)

    return {
        'distance_to_default': dd,
        'expected_default_frequency': edf,
        'default_point': default_point,
        'asset_value': asset_value,
        'asset_volatility': asset_volatility,
    }


# =============================================================================
# Credit Spread Models
# =============================================================================

def credit_spread_merton(
    leverage: float,
    asset_volatility: float,
    risk_free_rate: float,
    time_to_maturity: float
) -> float:
    """
    Credit Spread from Merton Model.

    Calculates theoretical credit spread given leverage and volatility.

    Args:
        leverage: Debt / Asset value (D/V)
        asset_volatility: Asset volatility
        risk_free_rate: Risk-free rate
        time_to_maturity: Time to debt maturity

    Returns:
        Credit spread in decimal
    """
    if leverage <= 0 or leverage >= 1:
        return 0.0

    T = time_to_maturity
    sigma = asset_volatility
    r = risk_free_rate

    # Quasi-debt ratio (present value)
    d = leverage * math.exp(-r * T)

    if d >= 1:
        return float('inf')

    d1 = (-math.log(d) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    # Risky debt value
    risky_debt_ratio = norm_cdf(d2) + (1 / d) * norm_cdf(-d1)

    # Credit spread
    if risky_debt_ratio > 0:
        spread = -math.log(risky_debt_ratio) / T
    else:
        spread = float('inf')

    return spread


def credit_spread_from_pd(
    probability_of_default: float,
    loss_given_default: float,
    risk_free_rate: float,
    time_horizon: float = 1.0
) -> float:
    """
    Credit Spread from PD and LGD.

    Simplified spread calculation assuming risk-neutral pricing.

    Spread ≈ PD × LGD (annualized)

    More precise:
    Spread = -ln(1 - PD × LGD) / T

    Args:
        probability_of_default: PD
        loss_given_default: LGD (0-1)
        risk_free_rate: Risk-free rate
        time_horizon: Time horizon in years

    Returns:
        Credit spread
    """
    expected_loss = probability_of_default * loss_given_default

    if expected_loss >= 1:
        return float('inf')

    # Continuous compounding adjustment
    spread = -math.log(1 - expected_loss) / time_horizon

    return spread


# =============================================================================
# Expected Loss and LGD
# =============================================================================

def expected_loss(
    probability_of_default: float,
    exposure_at_default: float,
    loss_given_default: float
) -> float:
    """
    Expected Loss.

    EL = PD × EAD × LGD

    Args:
        probability_of_default: PD
        exposure_at_default: EAD
        loss_given_default: LGD (0-1)

    Returns:
        Expected loss amount
    """
    return probability_of_default * exposure_at_default * loss_given_default


def unexpected_loss(
    probability_of_default: float,
    exposure_at_default: float,
    loss_given_default: float,
    confidence_level: float = 0.99
) -> float:
    """
    Unexpected Loss (Credit VaR).

    UL = EAD × LGD × [N^(-1)(conf) × σ(PD)]

    Simplified: UL ≈ EAD × LGD × √(PD × (1-PD)) × z_conf

    Args:
        probability_of_default: PD
        exposure_at_default: EAD
        loss_given_default: LGD
        confidence_level: Confidence level (e.g., 0.99)

    Returns:
        Unexpected loss amount
    """
    # Standard deviation of default indicator
    sigma_pd = math.sqrt(probability_of_default * (1 - probability_of_default))

    # Z-score for confidence level
    z = norm_ppf(confidence_level)

    return exposure_at_default * loss_given_default * sigma_pd * z


def loss_given_default_estimate(
    collateral_value: float,
    exposure: float,
    recovery_costs: float = 0.0,
    seniority: str = 'senior_secured'
) -> float:
    """
    Loss Given Default Estimation.

    LGD = 1 - Recovery Rate

    Args:
        collateral_value: Value of collateral
        exposure: Total exposure
        recovery_costs: Costs of recovery process
        seniority: Debt seniority level

    Returns:
        LGD estimate (0-1)
    """
    # Base recovery by seniority
    seniority_recovery = {
        'senior_secured': 0.65,
        'senior_unsecured': 0.45,
        'subordinated': 0.30,
        'junior': 0.15,
    }

    base_recovery = seniority_recovery.get(seniority, 0.40)

    # Collateral-adjusted recovery
    if exposure > 0:
        collateral_recovery = max(0, collateral_value - recovery_costs) / exposure
        recovery = min(1.0, max(base_recovery, collateral_recovery))
    else:
        recovery = base_recovery

    return 1 - recovery


# =============================================================================
# Credit Rating Mapping
# =============================================================================

def pd_to_rating(probability_of_default: float) -> str:
    """
    Map PD to indicative credit rating.

    Based on historical 1-year default rates by rating.

    Args:
        probability_of_default: 1-year PD

    Returns:
        Indicative rating
    """
    # Approximate 1-year PD by rating (based on historical averages)
    if probability_of_default < 0.0001:
        return "AAA"
    elif probability_of_default < 0.0003:
        return "AA+"
    elif probability_of_default < 0.0005:
        return "AA"
    elif probability_of_default < 0.001:
        return "AA-"
    elif probability_of_default < 0.002:
        return "A+"
    elif probability_of_default < 0.004:
        return "A"
    elif probability_of_default < 0.007:
        return "A-"
    elif probability_of_default < 0.012:
        return "BBB+"
    elif probability_of_default < 0.020:
        return "BBB"
    elif probability_of_default < 0.035:
        return "BBB-"
    elif probability_of_default < 0.060:
        return "BB+"
    elif probability_of_default < 0.100:
        return "BB"
    elif probability_of_default < 0.150:
        return "BB-"
    elif probability_of_default < 0.200:
        return "B+"
    elif probability_of_default < 0.300:
        return "B"
    elif probability_of_default < 0.400:
        return "B-"
    elif probability_of_default < 0.500:
        return "CCC"
    else:
        return "CC/C"


def rating_to_pd(rating: str) -> float:
    """
    Map credit rating to typical PD.

    Args:
        rating: Credit rating

    Returns:
        Typical 1-year PD
    """
    rating_pd_map = {
        "AAA": 0.00005,
        "AA+": 0.0002,
        "AA": 0.0004,
        "AA-": 0.0007,
        "A+": 0.0015,
        "A": 0.003,
        "A-": 0.005,
        "BBB+": 0.009,
        "BBB": 0.015,
        "BBB-": 0.027,
        "BB+": 0.045,
        "BB": 0.08,
        "BB-": 0.12,
        "B+": 0.17,
        "B": 0.25,
        "B-": 0.35,
        "CCC": 0.45,
        "CC": 0.55,
        "C": 0.65,
        "D": 1.0,
    }

    return rating_pd_map.get(rating.upper(), 0.05)


# =============================================================================
# Credit Scoring
# =============================================================================

def credit_score_model(
    leverage: float,
    interest_coverage: float,
    profitability: float,
    size: float,
    liquidity: float,
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Simple Credit Scoring Model.

    Combines financial ratios into a credit score.

    Args:
        leverage: Debt/Assets or Debt/Equity
        interest_coverage: EBIT/Interest
        profitability: ROA or Net Margin
        size: Log(Assets) or Log(Revenue)
        liquidity: Current Ratio
        weights: Optional custom weights

    Returns:
        Dictionary with score components and total
    """
    if weights is None:
        weights = {
            'leverage': -0.25,
            'coverage': 0.20,
            'profitability': 0.25,
            'size': 0.15,
            'liquidity': 0.15,
        }

    # Normalize inputs (sigmoid-style transformation)
    def sigmoid(x: float, center: float, scale: float) -> float:
        return 1 / (1 + math.exp(-scale * (x - center)))

    scores = {
        'leverage_score': sigmoid(-leverage, -0.5, 3),  # Lower leverage = higher score
        'coverage_score': sigmoid(interest_coverage, 3, 0.5),
        'profitability_score': sigmoid(profitability, 0.05, 20),
        'size_score': sigmoid(size, 20, 0.3),  # Assumes log(assets)
        'liquidity_score': sigmoid(liquidity, 1.5, 2),
    }

    # Weighted total
    total = (
        weights['leverage'] * scores['leverage_score'] +
        weights['coverage'] * scores['coverage_score'] +
        weights['profitability'] * scores['profitability_score'] +
        weights['size'] * scores['size_score'] +
        weights['liquidity'] * scores['liquidity_score']
    )

    # Scale to 0-100
    total_scaled = 50 + 50 * total

    scores['total_score'] = max(0, min(100, total_scaled))
    scores['implied_pd'] = max(0.0001, 0.5 * math.exp(-0.05 * scores['total_score']))

    return scores


# =============================================================================
# Credit Analysis Class
# =============================================================================

@dataclass
class CreditAnalysisResult:
    """Comprehensive credit analysis result."""
    merton_result: Optional[MertonModelResult]
    distance_to_default: float
    probability_of_default: float
    expected_loss: float
    credit_spread: float
    implied_rating: str
    risk_assessment: str


class CreditAnalysis:
    """
    Comprehensive Credit Analysis.

    Combines structural models, scoring, and risk metrics.
    """

    def __init__(
        self,
        equity_value: float,
        equity_volatility: float,
        total_debt: float,
        short_term_debt: float = 0.0,
        long_term_debt: float = 0.0,
        risk_free_rate: float = 0.05,
        time_horizon: float = 1.0
    ):
        """Initialize credit analysis."""
        self.equity_value = equity_value
        self.equity_volatility = equity_volatility
        self.total_debt = total_debt
        self.short_term_debt = short_term_debt or total_debt * 0.3
        self.long_term_debt = long_term_debt or total_debt * 0.7
        self.risk_free_rate = risk_free_rate
        self.time_horizon = time_horizon

    def run_merton_model(self) -> MertonModelResult:
        """Run Merton structural model."""
        return merton_model(
            self.equity_value,
            self.equity_volatility,
            self.total_debt,
            self.risk_free_rate,
            self.time_horizon
        )

    def calculate_edf(self) -> Dict[str, float]:
        """Calculate KMV-style EDF."""
        return expected_default_frequency(
            self.equity_value,
            self.equity_volatility,
            self.short_term_debt,
            self.long_term_debt,
            self.time_horizon
        )

    def estimate_expected_loss(
        self,
        lgd: float = 0.45
    ) -> Dict[str, float]:
        """
        Estimate expected and unexpected loss.

        Args:
            lgd: Loss given default assumption

        Returns:
            Dictionary with loss metrics
        """
        edf_result = self.calculate_edf()
        pd = edf_result['expected_default_frequency']
        ead = self.total_debt

        el = expected_loss(pd, ead, lgd)
        ul = unexpected_loss(pd, ead, lgd, 0.99)

        return {
            'expected_loss': el,
            'unexpected_loss_99': ul,
            'expected_loss_rate': el / ead if ead > 0 else 0,
            'pd': pd,
            'lgd': lgd,
            'ead': ead,
        }

    def full_analysis(self, lgd: float = 0.45) -> CreditAnalysisResult:
        """
        Perform comprehensive credit analysis.

        Args:
            lgd: Loss given default assumption

        Returns:
            Complete credit analysis result
        """
        # Run models
        merton = self.run_merton_model()
        edf_result = self.calculate_edf()

        # Get metrics
        dd = edf_result['distance_to_default']
        pd = edf_result['expected_default_frequency']
        loss = self.estimate_expected_loss(lgd)

        # Credit spread
        spread = credit_spread_from_pd(pd, lgd, self.risk_free_rate, self.time_horizon)

        # Implied rating
        rating = pd_to_rating(pd)

        # Risk assessment
        if pd < 0.005:
            assessment = "Low credit risk - Investment grade"
        elif pd < 0.02:
            assessment = "Moderate credit risk - Lower investment grade"
        elif pd < 0.08:
            assessment = "Elevated credit risk - Speculative grade"
        elif pd < 0.20:
            assessment = "High credit risk - Highly speculative"
        else:
            assessment = "Very high credit risk - Distressed"

        return CreditAnalysisResult(
            merton_result=merton,
            distance_to_default=dd,
            probability_of_default=pd,
            expected_loss=loss['expected_loss_rate'],
            credit_spread=spread,
            implied_rating=rating,
            risk_assessment=assessment
        )

    def sensitivity_analysis(
        self,
        equity_vol_range: Tuple[float, float] = (0.2, 0.6),
        leverage_range: Tuple[float, float] = (0.3, 0.7),
        steps: int = 5
    ) -> List[Dict]:
        """
        Sensitivity analysis of credit metrics.

        Args:
            equity_vol_range: Range of equity volatilities
            leverage_range: Range of debt/asset ratios
            steps: Number of steps in each dimension

        Returns:
            List of scenario results
        """
        results = []

        vol_step = (equity_vol_range[1] - equity_vol_range[0]) / (steps - 1)
        lev_step = (leverage_range[1] - leverage_range[0]) / (steps - 1)

        for i in range(steps):
            vol = equity_vol_range[0] + i * vol_step

            for j in range(steps):
                leverage = leverage_range[0] + j * lev_step

                # Adjust debt for leverage scenario
                asset_value = self.equity_value + self.total_debt
                scenario_debt = asset_value * leverage
                scenario_equity = asset_value * (1 - leverage)

                # Calculate EDF
                edf = expected_default_frequency(
                    scenario_equity,
                    vol,
                    scenario_debt * 0.3,
                    scenario_debt * 0.7,
                    self.time_horizon
                )

                results.append({
                    'equity_volatility': vol,
                    'leverage': leverage,
                    'distance_to_default': edf['distance_to_default'],
                    'probability_of_default': edf['expected_default_frequency'],
                    'implied_rating': pd_to_rating(edf['expected_default_frequency']),
                })

        return results
