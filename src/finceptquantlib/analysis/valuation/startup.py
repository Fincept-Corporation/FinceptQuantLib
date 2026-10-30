"""
Startup Valuation Models

Specialized valuation methods for early-stage, pre-revenue companies:
- Venture Capital Method
- Berkus Method
- Scorecard Valuation
- Risk Factor Summation
- First Chicago Method (scenario-based)
- Option Pricing Valuation
- Milestone-Based Valuation
- Network Effects Valuation

Used by VCs, angel investors, and startup founders.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Dict, List, Optional, Tuple
import math


class StartupStage(Enum):
    """Startup development stages."""
    IDEA = "idea"
    PROTOTYPE = "prototype"
    MVP = "mvp"
    EARLY_TRACTION = "early_traction"
    GROWTH = "growth"
    SCALE = "scale"


class RiskLevel(Enum):
    """Risk assessment levels."""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5


@dataclass
class ValuationResult:
    """
    Startup valuation result.

    Attributes:
        pre_money_valuation: Pre-money company valuation
        post_money_valuation: Post-money valuation (after investment)
        investment_amount: Investment size
        equity_percentage: Equity % given to investor
        price_per_share: Price per share (if applicable)
        method: Valuation method used
        assumptions: Key assumptions and parameters
    """
    pre_money_valuation: float
    post_money_valuation: float
    investment_amount: float = 0.0
    equity_percentage: float = 0.0
    price_per_share: Optional[float] = None
    method: str = ""
    assumptions: Dict[str, any] = None

    def __post_init__(self):
        if self.assumptions is None:
            self.assumptions = {}

        # Calculate derived values
        if self.investment_amount > 0 and self.pre_money_valuation > 0:
            self.post_money_valuation = self.pre_money_valuation + self.investment_amount
            self.equity_percentage = (self.investment_amount / self.post_money_valuation) * 100


class VentureCapitalMethod:
    """
    Venture Capital (VC) Method.

    Works backwards from expected exit to determine current valuation.

    Formula:
        Terminal Value = Revenue_exit * Revenue_Multiple
        OR Terminal Value = EBITDA_exit * EBITDA_Multiple

        Post-Money Valuation = Terminal Value / (1 + Required_ROI)^years_to_exit
        Pre-Money Valuation = Post-Money - Investment

    Examples:
        >>> vc = VentureCapitalMethod(
        ...     exit_value=100_000_000,
        ...     years_to_exit=5,
        ...     target_roi=10.0  # 10x return
        ... )
        >>> result = vc.calculate(investment=2_000_000)
        >>> print(f"Pre-money: ${result.pre_money_valuation:,.0f}")
        Pre-money: $8,000,000
    """

    def __init__(
        self,
        exit_value: float,
        years_to_exit: float,
        target_roi: float = 10.0,  # 10x return
        retention_ratio: float = 1.0  # Adjust for dilution
    ) -> None:
        """
        Initialize VC method.

        Args:
            exit_value: Expected exit value (acquisition/IPO)
            years_to_exit: Years until exit event
            target_roi: Target return multiple (e.g., 10 = 10x return)
            retention_ratio: Expected ownership retention (accounts for dilution)
        """
        self.exit_value = exit_value
        self.years_to_exit = years_to_exit
        self.target_roi = target_roi
        self.retention_ratio = retention_ratio

    def calculate(self, investment: float, shares_outstanding: Optional[int] = None) -> ValuationResult:
        """
        Calculate pre-money valuation.

        Args:
            investment: Investment amount
            shares_outstanding: Number of existing shares

        Returns:
            ValuationResult with valuation breakdown
        """
        # Post-money valuation
        post_money = self.exit_value / self.target_roi

        # Adjust for dilution
        post_money_adjusted = post_money * self.retention_ratio

        # Pre-money
        pre_money = post_money_adjusted - investment

        # Ownership %
        ownership_pct = (investment / post_money_adjusted) * 100

        # Price per share
        price_per_share = None
        if shares_outstanding:
            new_shares = (investment / (post_money_adjusted / shares_outstanding))
            price_per_share = investment / new_shares

        return ValuationResult(
            pre_money_valuation=pre_money,
            post_money_valuation=post_money_adjusted,
            investment_amount=investment,
            equity_percentage=ownership_pct,
            price_per_share=price_per_share,
            method="Venture Capital Method",
            assumptions={
                "exit_value": self.exit_value,
                "years_to_exit": self.years_to_exit,
                "target_roi": self.target_roi,
                "retention_ratio": self.retention_ratio
            }
        )

    @classmethod
    def from_revenue_multiple(
        cls,
        projected_revenue: float,
        revenue_multiple: float,
        years_to_exit: float,
        target_roi: float = 10.0
    ) -> VentureCapitalMethod:
        """
        Create from revenue projections.

        Args:
            projected_revenue: Revenue at exit
            revenue_multiple: Industry revenue multiple
            years_to_exit: Years to exit
            target_roi: Target ROI multiple

        Returns:
            VentureCapitalMethod instance
        """
        exit_value = projected_revenue * revenue_multiple
        return cls(exit_value, years_to_exit, target_roi)


@dataclass
class BerkusScore:
    """Berkus Method scoring components."""
    sound_idea: float = 0.0  # Max $500k
    prototype: float = 0.0  # Max $500k
    quality_management: float = 0.0  # Max $500k
    strategic_relationships: float = 0.0  # Max $500k
    product_rollout: float = 0.0  # Max $500k

    def total_valuation(self) -> float:
        """Calculate total Berkus valuation."""
        return (self.sound_idea + self.prototype + self.quality_management +
                self.strategic_relationships + self.product_rollout)


class BerkusMethod:
    """
    Berkus Method for pre-revenue valuation.

    Assigns value to 5 qualitative success factors.
    Typical range: $0 - $2.5M (max $500k per factor)

    Examples:
        >>> berkus = BerkusMethod()
        >>> berkus.set_score(sound_idea=400_000, prototype=300_000,
        ...                  quality_management=500_000, strategic_relationships=200_000)
        >>> result = berkus.calculate()
        >>> print(f"Valuation: ${result.pre_money_valuation:,.0f}")
        Valuation: $1,400,000
    """

    MAX_PER_FACTOR = 500_000

    def __init__(self) -> None:
        """Initialize Berkus method."""
        self.scores = BerkusScore()

    def set_score(
        self,
        sound_idea: float = 0.0,
        prototype: float = 0.0,
        quality_management: float = 0.0,
        strategic_relationships: float = 0.0,
        product_rollout: float = 0.0
    ) -> None:
        """
        Set scores for each factor.

        Args:
            sound_idea: Value for idea/concept quality (max $500k)
            prototype: Value for working prototype (max $500k)
            quality_management: Value for team quality (max $500k)
            strategic_relationships: Value for partnerships (max $500k)
            product_rollout: Value for sales/distribution (max $500k)
        """
        self.scores = BerkusScore(
            sound_idea=min(sound_idea, self.MAX_PER_FACTOR),
            prototype=min(prototype, self.MAX_PER_FACTOR),
            quality_management=min(quality_management, self.MAX_PER_FACTOR),
            strategic_relationships=min(strategic_relationships, self.MAX_PER_FACTOR),
            product_rollout=min(product_rollout, self.MAX_PER_FACTOR)
        )

    def calculate(self) -> ValuationResult:
        """Calculate valuation using Berkus method."""
        valuation = self.scores.total_valuation()

        return ValuationResult(
            pre_money_valuation=valuation,
            post_money_valuation=valuation,
            method="Berkus Method",
            assumptions={
                "sound_idea": self.scores.sound_idea,
                "prototype": self.scores.prototype,
                "quality_management": self.scores.quality_management,
                "strategic_relationships": self.scores.strategic_relationships,
                "product_rollout": self.scores.product_rollout
            }
        )


@dataclass
class ScorecardFactors:
    """Scorecard method comparison factors."""
    management_team: float = 0.0  # Weight: 0-30%
    size_of_opportunity: float = 0.0  # Weight: 0-25%
    product_technology: float = 0.0  # Weight: 0-15%
    competitive_environment: float = 0.0  # Weight: 0-10%
    marketing_sales: float = 0.0  # Weight: 0-10%
    need_for_funding: float = 0.0  # Weight: 0-5%
    other_factors: float = 0.0  # Weight: 0-5%

    def weighted_multiplier(self) -> float:
        """
        Calculate weighted adjustment multiplier.

        Each factor is scored relative to comparables (0.5 - 1.5 multiplier).
        """
        weights = {
            'management_team': 0.30,
            'size_of_opportunity': 0.25,
            'product_technology': 0.15,
            'competitive_environment': 0.10,
            'marketing_sales': 0.10,
            'need_for_funding': 0.05,
            'other_factors': 0.05
        }

        total_adjustment = 0.0
        for factor, weight in weights.items():
            score = getattr(self, factor)
            total_adjustment += (score - 1.0) * weight

        return 1.0 + total_adjustment


class ScorecardValuation:
    """
    Scorecard Valuation Method.

    Compares target startup to recently funded comparable startups.

    Examples:
        >>> scorecard = ScorecardValuation(
        ...     comparable_avg_valuation=3_000_000,
        ...     region="US-California",
        ...     industry="SaaS"
        ... )
        >>> scorecard.set_factors(management_team=1.2, size_of_opportunity=1.3,
        ...                       product_technology=1.1)
        >>> result = scorecard.calculate()
    """

    def __init__(
        self,
        comparable_avg_valuation: float,
        region: str = "US",
        industry: str = "Technology"
    ) -> None:
        """
        Initialize scorecard method.

        Args:
            comparable_avg_valuation: Average pre-money valuation of comparables
            region: Geographic region
            industry: Industry sector
        """
        self.base_valuation = comparable_avg_valuation
        self.region = region
        self.industry = industry
        self.factors = ScorecardFactors()

    def set_factors(
        self,
        management_team: float = 1.0,
        size_of_opportunity: float = 1.0,
        product_technology: float = 1.0,
        competitive_environment: float = 1.0,
        marketing_sales: float = 1.0,
        need_for_funding: float = 1.0,
        other_factors: float = 1.0
    ) -> None:
        """
        Set comparison factors.

        Args:
            Each factor: Multiplier relative to comparables
                1.0 = same as comparables
                <1.0 = worse than comparables
                >1.0 = better than comparables
                Typical range: 0.5 - 1.5
        """
        self.factors = ScorecardFactors(
            management_team=management_team,
            size_of_opportunity=size_of_opportunity,
            product_technology=product_technology,
            competitive_environment=competitive_environment,
            marketing_sales=marketing_sales,
            need_for_funding=need_for_funding,
            other_factors=other_factors
        )

    def calculate(self) -> ValuationResult:
        """Calculate scorecard-adjusted valuation."""
        multiplier = self.factors.weighted_multiplier()
        valuation = self.base_valuation * multiplier

        return ValuationResult(
            pre_money_valuation=valuation,
            post_money_valuation=valuation,
            method="Scorecard Valuation Method",
            assumptions={
                "base_valuation": self.base_valuation,
                "adjustment_multiplier": multiplier,
                "region": self.region,
                "industry": self.industry,
                **vars(self.factors)
            }
        )


class RiskFactorSummation:
    """
    Risk Factor Summation Method.

    Starts with average valuation and adjusts for 12 risk factors.
    Each factor: +/- $250k based on risk level.

    Examples:
        >>> rfs = RiskFactorSummation(base_valuation=2_000_000)
        >>> rfs.assess_risks(
        ...     management=RiskLevel.LOW,
        ...     technology=RiskLevel.MEDIUM,
        ...     market=RiskLevel.HIGH
        ... )
        >>> result = rfs.calculate()
    """

    RISK_ADJUSTMENT = 250_000  # Per factor

    # 12 Standard Risk Factors
    RISK_FACTORS = [
        "management",
        "stage_of_business",
        "political_legal",
        "manufacturing",
        "sales_marketing",
        "funding_capital_raising",
        "competition",
        "technology",
        "litigation",
        "international",
        "reputation",
        "exit_potential"
    ]

    def __init__(self, base_valuation: float = 2_000_000) -> None:
        """
        Initialize risk factor summation.

        Args:
            base_valuation: Starting valuation (typical $2M for pre-revenue)
        """
        self.base_valuation = base_valuation
        self.risk_assessments: Dict[str, RiskLevel] = dict.fromkeys(self.RISK_FACTORS, RiskLevel.MEDIUM)

    def assess_risks(self, **risk_levels: RiskLevel) -> None:
        """
        Assess risk levels for each factor.

        Args:
            **risk_levels: Risk factor names and their RiskLevel
                Valid factors: management, technology, market, competition, etc.
        """
        for factor, level in risk_levels.items():
            if factor in self.risk_assessments:
                self.risk_assessments[factor] = level

    def calculate(self) -> ValuationResult:
        """
        Calculate valuation with risk adjustments.

        Returns:
            ValuationResult
        """
        total_adjustment = 0.0

        for factor, risk_level in self.risk_assessments.items():
            # Risk adjustment: MEDIUM = 0, LOW/VERY_LOW = +, HIGH/VERY_HIGH = -
            adjustment_factor = (RiskLevel.MEDIUM.value - risk_level.value)
            total_adjustment += adjustment_factor * self.RISK_ADJUSTMENT

        valuation = self.base_valuation + total_adjustment

        # Ensure non-negative
        valuation = max(valuation, 0)

        return ValuationResult(
            pre_money_valuation=valuation,
            post_money_valuation=valuation,
            method="Risk Factor Summation Method",
            assumptions={
                "base_valuation": self.base_valuation,
                "total_risk_adjustment": total_adjustment,
                "risk_assessments": {k: v.name for k, v in self.risk_assessments.items()}
            }
        )


@dataclass
class ValuationScenario:
    """Scenario for First Chicago method."""
    name: str
    probability: float
    revenue: float
    ebitda_margin: float
    exit_multiple: float
    years_to_exit: float

    def terminal_value(self) -> float:
        """Calculate terminal value for this scenario."""
        ebitda = self.revenue * self.ebitda_margin
        return ebitda * self.exit_multiple

    def present_value(self, discount_rate: float) -> float:
        """Discount terminal value to present."""
        tv = self.terminal_value()
        return tv / ((1 + discount_rate) ** self.years_to_exit)


class FirstChicagoMethod:
    """
    First Chicago Method (Scenario-Based Valuation).

    Uses probability-weighted scenarios:
    - Best case (30%)
    - Base case (50%)
    - Worst case (20%)

    Examples:
        >>> fc = FirstChicagoMethod(discount_rate=0.30)
        >>> fc.add_scenario("Best", 0.30, revenue=50_000_000, ebitda_margin=0.25,
        ...                 exit_multiple=12, years_to_exit=5)
        >>> fc.add_scenario("Base", 0.50, revenue=20_000_000, ebitda_margin=0.20,
        ...                 exit_multiple=8, years_to_exit=5)
        >>> fc.add_scenario("Worst", 0.20, revenue=5_000_000, ebitda_margin=0.10,
        ...                 exit_multiple=5, years_to_exit=5)
        >>> result = fc.calculate()
    """

    def __init__(self, discount_rate: float = 0.30) -> None:
        """
        Initialize First Chicago method.

        Args:
            discount_rate: Discount rate for startup risk (typically 25-40%)
        """
        self.discount_rate = discount_rate
        self.scenarios: List[ValuationScenario] = []

    def add_scenario(
        self,
        name: str,
        probability: float,
        revenue: float,
        ebitda_margin: float,
        exit_multiple: float,
        years_to_exit: float
    ) -> None:
        """Add a scenario."""
        scenario = ValuationScenario(
            name=name,
            probability=probability,
            revenue=revenue,
            ebitda_margin=ebitda_margin,
            exit_multiple=exit_multiple,
            years_to_exit=years_to_exit
        )
        self.scenarios.append(scenario)

    def calculate(self) -> ValuationResult:
        """
        Calculate probability-weighted valuation.

        Returns:
            ValuationResult
        """
        if not self.scenarios:
            raise ValueError("No scenarios added")

        # Probability-weighted PV
        weighted_value = 0.0
        for scenario in self.scenarios:
            pv = scenario.present_value(self.discount_rate)
            weighted_value += scenario.probability * pv

        return ValuationResult(
            pre_money_valuation=weighted_value,
            post_money_valuation=weighted_value,
            method="First Chicago Method",
            assumptions={
                "discount_rate": self.discount_rate,
                "num_scenarios": len(self.scenarios),
                "scenarios": [
                    {
                        "name": s.name,
                        "probability": s.probability,
                        "terminal_value": s.terminal_value(),
                        "present_value": s.present_value(self.discount_rate)
                    }
                    for s in self.scenarios
                ]
            }
        )


class OptionPricingValuation:
    """
    Option Pricing Method for startup valuation.

    Treats startup as a call option:
    - Underlying asset = Total Addressable Market (TAM)
    - Strike price = Cost to capture market
    - Time to expiration = Runway to profitability
    - Volatility = Market/technology uncertainty

    Uses Black-Scholes framework adapted for startups.

    Examples:
        >>> option_val = OptionPricingValuation(
        ...     total_addressable_market=1_000_000_000,
        ...     cost_to_capture=50_000_000,
        ...     time_to_profitability=3.0,
        ...     volatility=0.80,
        ...     risk_free_rate=0.03
        ... )
        >>> result = option_val.calculate(market_share_capture=0.10)
    """

    def __init__(
        self,
        total_addressable_market: float,
        cost_to_capture: float,
        time_to_profitability: float,
        volatility: float = 0.80,
        risk_free_rate: float = 0.03
    ) -> None:
        """
        Initialize option pricing valuation.

        Args:
            total_addressable_market: TAM size
            cost_to_capture: Total investment needed
            time_to_profitability: Years to profitability
            volatility: Market volatility (typically 60-100% for startups)
            risk_free_rate: Risk-free rate
        """
        self.tam = total_addressable_market
        self.strike = cost_to_capture
        self.time = time_to_profitability
        self.volatility = volatility
        self.risk_free_rate = risk_free_rate

    def calculate(self, market_share_capture: float = 0.05) -> ValuationResult:
        """
        Calculate option value (startup valuation).

        Args:
            market_share_capture: Expected market share % (as decimal)

        Returns:
            ValuationResult
        """
        from scipy.stats import norm

        # Underlying = TAM * market share
        S = self.tam * market_share_capture
        K = self.strike
        T = self.time
        sigma = self.volatility
        r = self.risk_free_rate

        # Black-Scholes call option formula
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        call_value = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

        return ValuationResult(
            pre_money_valuation=call_value,
            post_money_valuation=call_value,
            method="Option Pricing Valuation",
            assumptions={
                "tam": self.tam,
                "market_share": market_share_capture,
                "underlying_value": S,
                "strike_cost": K,
                "time_to_profitability": T,
                "volatility": sigma,
                "risk_free_rate": r,
                "d1": d1,
                "d2": d2
            }
        )


@dataclass
class Milestone:
    """Development milestone with value."""
    name: str
    description: str
    target_date: date
    completion_percentage: float  # 0-100%
    value_created: float
    probability_of_achievement: float = 1.0


class MilestoneValuation:
    """
    Milestone-Based Valuation.

    Values company based on achievement of key milestones:
    - Technical: Prototype, MVP, Beta, Launch
    - Business: First customer, $1M ARR, Profitability
    - Funding: Seed, Series A, B, etc.

    Examples:
        >>> milestone_val = MilestoneValuation(base_value=500_000)
        >>> milestone_val.add_milestone("MVP Launch", value_created=1_000_000,
        ...                             completion=0.75, probability=0.90)
        >>> milestone_val.add_milestone("First Revenue", value_created=2_000_000,
        ...                             completion=0.30, probability=0.70)
        >>> result = milestone_val.calculate()
    """

    def __init__(self, base_value: float = 500_000) -> None:
        """
        Initialize milestone valuation.

        Args:
            base_value: Base value with no milestones
        """
        self.base_value = base_value
        self.milestones: List[Milestone] = []

    def add_milestone(
        self,
        name: str,
        value_created: float,
        completion: float = 0.0,
        probability: float = 1.0,
        description: str = "",
        target_date: Optional[date] = None
    ) -> None:
        """
        Add a milestone.

        Args:
            name: Milestone name
            value_created: Value added when achieved
            completion: Completion percentage (0-100)
            probability: Probability of achieving (0-1)
            description: Description
            target_date: Expected completion date
        """
        milestone = Milestone(
            name=name,
            description=description,
            target_date=target_date or date.today(),
            completion_percentage=completion,
            value_created=value_created,
            probability_of_achievement=probability
        )
        self.milestones.append(milestone)

    def calculate(self, risk_adjustment: float = 0.8) -> ValuationResult:
        """
        Calculate valuation based on milestones.

        Args:
            risk_adjustment: Risk discount factor (0-1)

        Returns:
            ValuationResult
        """
        milestone_value = 0.0

        for milestone in self.milestones:
            # Value = Created Value * Completion% * Probability * Risk Adjustment
            contrib = (milestone.value_created *
                      (milestone.completion_percentage / 100) *
                      milestone.probability_of_achievement *
                      risk_adjustment)
            milestone_value += contrib

        total_valuation = self.base_value + milestone_value

        return ValuationResult(
            pre_money_valuation=total_valuation,
            post_money_valuation=total_valuation,
            method="Milestone-Based Valuation",
            assumptions={
                "base_value": self.base_value,
                "milestone_value": milestone_value,
                "risk_adjustment": risk_adjustment,
                "num_milestones": len(self.milestones),
                "milestones": [
                    {
                        "name": m.name,
                        "completion": m.completion_percentage,
                        "probability": m.probability_of_achievement,
                        "value": m.value_created
                    }
                    for m in self.milestones
                ]
            }
        )


class NetworkEffectsValuation:
    """
    Network Effects Valuation.

    For platforms where value scales with network size.
    Uses Metcalfe's Law: Value ∝ n² (or Reed's Law: Value ∝ 2^n)

    Examples:
        >>> network_val = NetworkEffectsValuation(
        ...     current_users=10_000,
        ...     value_per_user=50,
        ...     network_coefficient=1.5
        ... )
        >>> result = network_val.calculate(projected_users_1yr=100_000)
    """

    def __init__(
        self,
        current_users: int,
        value_per_user: float,
        network_coefficient: float = 2.0,  # Metcalfe's law
        viral_coefficient: float = 1.0
    ) -> None:
        """
        Initialize network effects valuation.

        Args:
            current_users: Current user base
            value_per_user: Base value per user
            network_coefficient: Network effect strength (1.0-2.0 for Metcalfe)
            viral_coefficient: Viral growth factor (k-factor)
        """
        self.current_users = current_users
        self.value_per_user = value_per_user
        self.network_coefficient = network_coefficient
        self.viral_coefficient = viral_coefficient

    def calculate(
        self,
        projected_users_1yr: int,
        projected_users_3yr: Optional[int] = None,
        discount_rate: float = 0.30
    ) -> ValuationResult:
        """
        Calculate valuation with network effects.

        Args:
            projected_users_1yr: Projected users in 1 year
            projected_users_3yr: Projected users in 3 years
            discount_rate: Discount rate

        Returns:
            ValuationResult
        """
        # Current network value
        current_network_value = self._network_value(self.current_users)

        # Future network value
        future_value_1yr = self._network_value(projected_users_1yr)
        pv_1yr = future_value_1yr / ((1 + discount_rate) ** 1)

        # 3-year projection if provided
        if projected_users_3yr:
            future_value_3yr = self._network_value(projected_users_3yr)
            pv_3yr = future_value_3yr / ((1 + discount_rate) ** 3)
            valuation = (current_network_value + pv_1yr + pv_3yr) / 3
        else:
            valuation = (current_network_value + pv_1yr) / 2

        return ValuationResult(
            pre_money_valuation=valuation,
            post_money_valuation=valuation,
            method="Network Effects Valuation",
            assumptions={
                "current_users": self.current_users,
                "projected_users_1yr": projected_users_1yr,
                "projected_users_3yr": projected_users_3yr,
                "value_per_user": self.value_per_user,
                "network_coefficient": self.network_coefficient,
                "current_network_value": current_network_value,
                "discount_rate": discount_rate
            }
        )

    def _network_value(self, num_users: int) -> float:
        """
        Calculate network value.

        Metcalfe's Law: V = k * n^2
        """
        return self.value_per_user * (num_users ** self.network_coefficient)


def pre_revenue_valuation_range(
    stage: StartupStage,
    sector: str = "Technology",
    region: str = "US"
) -> Tuple[float, float]:
    """
    Get typical pre-revenue valuation range.

    Args:
        stage: Startup development stage
        sector: Industry sector
        region: Geographic region

    Returns:
        (low, high) valuation range
    """
    # Typical ranges by stage (US technology startups, 2024)
    ranges = {
        StartupStage.IDEA: (250_000, 1_000_000),
        StartupStage.PROTOTYPE: (500_000, 2_000_000),
        StartupStage.MVP: (1_000_000, 3_000_000),
        StartupStage.EARLY_TRACTION: (2_000_000, 8_000_000),
        StartupStage.GROWTH: (5_000_000, 20_000_000),
        StartupStage.SCALE: (10_000_000, 50_000_000),
    }

    base_low, base_high = ranges.get(stage, (1_000_000, 5_000_000))

    # Sector adjustments
    sector_multipliers = {
        "AI/ML": 1.5,
        "Biotech": 1.3,
        "SaaS": 1.2,
        "Fintech": 1.2,
        "E-commerce": 0.9,
        "Technology": 1.0
    }
    multiplier = sector_multipliers.get(sector, 1.0)

    # Region adjustments
    region_multipliers = {
        "US-Silicon Valley": 1.5,
        "US-New York": 1.3,
        "US-Other": 1.0,
        "Europe": 0.8,
        "Asia": 0.7,
        "Other": 0.6
    }
    region_mult = region_multipliers.get(region, 1.0)

    final_low = base_low * multiplier * region_mult
    final_high = base_high * multiplier * region_mult

    return (final_low, final_high)


def calculate_ownership_dilution(
    pre_money: float,
    investment: float,
    option_pool_pct: float = 0.15
) -> Dict[str, float]:
    """
    Calculate ownership dilution from investment.

    Args:
        pre_money: Pre-money valuation
        investment: Investment amount
        option_pool_pct: Employee option pool % (pre-dilution)

    Returns:
        Dictionary with ownership percentages
    """
    # Post-money (before option pool)
    post_money_pre_pool = pre_money + investment

    # Option pool dilutes everyone
    total_post_money = post_money_pre_pool / (1 - option_pool_pct)

    # Ownership calculations
    founder_pct = (pre_money / total_post_money) * 100
    investor_pct = (investment / total_post_money) * 100
    option_pool = option_pool_pct * 100

    return {
        "founders": founder_pct,
        "investors": investor_pct,
        "option_pool": option_pool,
        "total": founder_pct + investor_pct + option_pool,
        "post_money_valuation": total_post_money
    }
