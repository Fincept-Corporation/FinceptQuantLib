"""
Stress Testing and Scenario Analysis

Provides tools for:
- Historical scenario analysis (2008 crisis, COVID, etc.)
- Hypothetical stress scenarios
- Factor-based stress testing
- Reverse stress testing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional, List, Dict, Callable
from enum import Enum


class ScenarioType(Enum):
    """Type of stress scenario."""
    HISTORICAL = "historical"
    HYPOTHETICAL = "hypothetical"
    REGULATORY = "regulatory"
    CUSTOM = "custom"


@dataclass
class Scenario:
    """
    Definition of a stress scenario.

    A scenario specifies shocks to various risk factors.
    """
    name: str
    description: str
    scenario_type: ScenarioType
    shocks: Dict[str, float]  # {risk_factor: shock_value}
    probability: Optional[float] = None
    severity: Optional[str] = None
    date_reference: Optional[date] = None

    def __repr__(self) -> str:
        return f"Scenario({self.name}, type={self.scenario_type.value}, factors={len(self.shocks)})"


@dataclass
class StressResult:
    """Result of stress testing."""
    scenario_name: str
    base_value: float
    stressed_value: float
    pnl_impact: float
    pnl_impact_pct: float
    factor_contributions: Dict[str, float] = field(default_factory=dict)
    breaches: List[str] = field(default_factory=list)


# Pre-defined historical scenarios
HISTORICAL_SCENARIOS = {
    "2008_crisis": Scenario(
        name="2008 Financial Crisis",
        description="Lehman Brothers collapse and credit crisis",
        scenario_type=ScenarioType.HISTORICAL,
        shocks={
            "equity": -0.40,          # 40% equity decline
            "credit_spread": 0.03,    # 300bp credit spread widening
            "ir_curve": -0.01,        # 100bp rate decline
            "volatility": 0.50,       # 50% vol increase
            "fx_usd": 0.10,           # 10% USD appreciation
        },
        date_reference=date(2008, 9, 15),
        severity="severe",
    ),
    "covid_2020": Scenario(
        name="COVID-19 March 2020",
        description="COVID-19 market crash",
        scenario_type=ScenarioType.HISTORICAL,
        shocks={
            "equity": -0.34,
            "credit_spread": 0.025,
            "ir_curve": -0.015,
            "volatility": 0.60,
            "oil": -0.65,
        },
        date_reference=date(2020, 3, 23),
        severity="severe",
    ),
    "dot_com_2000": Scenario(
        name="Dot-Com Bubble Burst",
        description="Technology bubble collapse",
        scenario_type=ScenarioType.HISTORICAL,
        shocks={
            "equity": -0.49,
            "tech_equity": -0.78,
            "ir_curve": -0.02,
            "volatility": 0.30,
        },
        date_reference=date(2000, 3, 10),
        severity="severe",
    ),
    "eu_debt_2011": Scenario(
        name="European Debt Crisis",
        description="Eurozone sovereign debt crisis",
        scenario_type=ScenarioType.HISTORICAL,
        shocks={
            "equity": -0.25,
            "credit_spread": 0.02,
            "eu_sovereign_spread": 0.05,
            "fx_eur": -0.15,
            "volatility": 0.25,
        },
        date_reference=date(2011, 8, 8),
        severity="high",
    ),
    "rate_shock_up": Scenario(
        name="Rapid Rate Increase",
        description="200bp parallel rate increase",
        scenario_type=ScenarioType.HYPOTHETICAL,
        shocks={
            "ir_1m": 0.02,
            "ir_3m": 0.02,
            "ir_6m": 0.02,
            "ir_1y": 0.02,
            "ir_2y": 0.02,
            "ir_5y": 0.02,
            "ir_10y": 0.02,
            "ir_30y": 0.02,
            "equity": -0.10,
            "credit_spread": 0.005,
        },
        severity="high",
    ),
    "curve_steepening": Scenario(
        name="Yield Curve Steepening",
        description="Short rates fall, long rates rise",
        scenario_type=ScenarioType.HYPOTHETICAL,
        shocks={
            "ir_1m": -0.01,
            "ir_3m": -0.01,
            "ir_6m": -0.005,
            "ir_1y": 0.0,
            "ir_2y": 0.005,
            "ir_5y": 0.01,
            "ir_10y": 0.015,
            "ir_30y": 0.02,
        },
        severity="moderate",
    ),
    "credit_crisis": Scenario(
        name="Credit Crisis",
        description="Major credit spread widening",
        scenario_type=ScenarioType.HYPOTHETICAL,
        shocks={
            "credit_spread_ig": 0.01,
            "credit_spread_hy": 0.05,
            "equity": -0.15,
            "volatility": 0.30,
            "liquidity": -0.20,
        },
        severity="severe",
    ),
}


class StressTest:
    """
    Stress Testing Framework.

    Applies stress scenarios to portfolios and calculates impact.
    """

    def __init__(
        self,
        base_value: float,
        pricing_func: Optional[Callable[[Dict[str, float]], float]] = None,
        factor_sensitivities: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize stress tester.

        Args:
            base_value: Current portfolio value
            pricing_func: Function that reprices portfolio given shocks
            factor_sensitivities: Dict of {factor: delta} for linear approximation
        """
        self.base_value = base_value
        self.pricing_func = pricing_func
        self.factor_sensitivities = factor_sensitivities or {}

    def run_scenario(
        self,
        scenario: Scenario,
        use_full_reval: bool = True,
    ) -> StressResult:
        """
        Run a single stress scenario.

        Args:
            scenario: Scenario to run
            use_full_reval: If True and pricing_func provided, do full revaluation

        Returns:
            StressResult
        """
        factor_contributions = {}

        if use_full_reval and self.pricing_func is not None:
            # Full revaluation
            stressed_value = self.pricing_func(scenario.shocks)
        else:
            # Linear approximation using sensitivities
            stressed_value = self.base_value
            for factor, shock in scenario.shocks.items():
                if factor in self.factor_sensitivities:
                    impact = self.factor_sensitivities[factor] * shock
                    factor_contributions[factor] = impact
                    stressed_value += impact

        pnl = stressed_value - self.base_value
        pnl_pct = pnl / self.base_value if self.base_value != 0 else 0

        return StressResult(
            scenario_name=scenario.name,
            base_value=self.base_value,
            stressed_value=stressed_value,
            pnl_impact=pnl,
            pnl_impact_pct=pnl_pct,
            factor_contributions=factor_contributions,
        )

    def run_all_scenarios(
        self,
        scenarios: Optional[List[Scenario]] = None,
    ) -> List[StressResult]:
        """
        Run multiple stress scenarios.

        Args:
            scenarios: List of scenarios (uses predefined if None)

        Returns:
            List of StressResult
        """
        if scenarios is None:
            scenarios = list(HISTORICAL_SCENARIOS.values())

        return [self.run_scenario(s) for s in scenarios]

    def worst_case_analysis(
        self,
        scenarios: Optional[List[Scenario]] = None,
    ) -> StressResult:
        """
        Find worst case scenario.

        Args:
            scenarios: List of scenarios to test

        Returns:
            StressResult for worst scenario
        """
        results = self.run_all_scenarios(scenarios)
        return min(results, key=lambda r: r.pnl_impact)

    def factor_shock_ladder(
        self,
        factor: str,
        shock_range: list[float],
    ) -> Dict[float, float]:
        """
        Generate shock ladder for a single factor.

        Args:
            factor: Risk factor to shock
            shock_range: Array of shock values

        Returns:
            Dict of {shock: pnl_impact}
        """
        results = {}

        for shock in shock_range:
            scenario = Scenario(
                name=f"{factor}_shock_{shock}",
                description=f"Shock {factor} by {shock}",
                scenario_type=ScenarioType.CUSTOM,
                shocks={factor: shock},
            )
            result = self.run_scenario(scenario)
            results[shock] = result.pnl_impact

        return results


class ScenarioAnalysis:
    """
    Comprehensive Scenario Analysis.

    Provides tools for building, running, and analyzing scenarios.
    """

    def __init__(
        self,
        base_value: float,
        factor_deltas: Dict[str, float],
        factor_gammas: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize scenario analyzer.

        Args:
            base_value: Base portfolio value
            factor_deltas: First-order sensitivities
            factor_gammas: Second-order sensitivities (optional)
        """
        self.base_value = base_value
        self.factor_deltas = factor_deltas
        self.factor_gammas = factor_gammas or {}

    def calculate_impact(
        self,
        shocks: Dict[str, float],
        include_gamma: bool = True,
    ) -> float:
        """
        Calculate P&L impact of shocks.

        Uses Taylor expansion:
        ΔV ≈ Σ δᵢ * Δxᵢ + ½ Σ γᵢ * Δxᵢ²

        Args:
            shocks: Dict of {factor: shock}
            include_gamma: Include second-order effects

        Returns:
            P&L impact
        """
        pnl = 0.0

        for factor, shock in shocks.items():
            # Delta effect
            if factor in self.factor_deltas:
                pnl += self.factor_deltas[factor] * shock

            # Gamma effect
            if include_gamma and factor in self.factor_gammas:
                pnl += 0.5 * self.factor_gammas[factor] * shock ** 2

        return pnl

    def decompose_pnl(
        self,
        shocks: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Decompose P&L by risk factor.

        Args:
            shocks: Scenario shocks

        Returns:
            Dict of {factor: pnl_contribution}
        """
        contributions = {}

        for factor, shock in shocks.items():
            contrib = 0.0
            if factor in self.factor_deltas:
                contrib += self.factor_deltas[factor] * shock
            if factor in self.factor_gammas:
                contrib += 0.5 * self.factor_gammas[factor] * shock ** 2
            contributions[factor] = contrib

        return contributions

    def reverse_stress(
        self,
        target_loss: float,
        factors: Optional[List[str]] = None,
        max_shock: float = 0.5,
    ) -> Dict[str, float]:
        """
        Reverse stress test: find shocks that produce target loss.

        Simplified: assumes equal relative shocks to all factors.

        Args:
            target_loss: Target loss amount
            factors: Factors to include (all if None)
            max_shock: Maximum shock magnitude

        Returns:
            Dict of {factor: required_shock}
        """
        if factors is None:
            factors = list(self.factor_deltas.keys())

        # Simple approach: binary search for scaling factor
        total_delta = sum(abs(self.factor_deltas.get(f, 0)) for f in factors)

        if total_delta == 0:
            return dict.fromkeys(factors, 0.0)

        # Find scaling
        required_scaling = -target_loss / total_delta

        # Cap at max_shock
        scaling = min(abs(required_scaling), max_shock)
        sign = 1 if required_scaling > 0 else (-1 if required_scaling < 0 else 0)

        def sign_of(x: float) -> int:
            return (x > 0) - (x < 0)

        return {f: sign * scaling * sign_of(self.factor_deltas.get(f, 0)) for f in factors}

    def sensitivity_matrix(
        self,
        factors: List[str],
        shock_sizes: List[float],
    ) -> list[list[float]]:
        """
        Generate sensitivity matrix.

        Args:
            factors: List of risk factors
            shock_sizes: List of shock sizes to test

        Returns:
            Matrix of P&L impacts [n_factors x n_shocks]
        """
        n_factors = len(factors)
        n_shocks = len(shock_sizes)
        matrix = [[0.0] * n_shocks for _ in range(n_factors)]

        for i, factor in enumerate(factors):
            for j, shock in enumerate(shock_sizes):
                matrix[i][j] = self.calculate_impact({factor: shock})

        return matrix


def build_custom_scenario(
    name: str,
    description: str,
    **factor_shocks: float,
) -> Scenario:
    """
    Build a custom scenario.

    Args:
        name: Scenario name
        description: Scenario description
        **factor_shocks: Factor shocks as keyword arguments

    Returns:
        Scenario

    Example:
        scenario = build_custom_scenario(
            "My Scenario",
            "Custom test scenario",
            equity=-0.20,
            credit_spread=0.01,
            ir_curve=-0.005,
        )
    """
    return Scenario(
        name=name,
        description=description,
        scenario_type=ScenarioType.CUSTOM,
        shocks=factor_shocks,
    )


def combine_scenarios(
    scenarios: List[Scenario],
    weights: Optional[List[float]] = None,
    name: str = "Combined Scenario",
) -> Scenario:
    """
    Combine multiple scenarios into one.

    Args:
        scenarios: List of scenarios to combine
        weights: Weights for each scenario (equal if None)
        name: Name for combined scenario

    Returns:
        Combined Scenario
    """
    if weights is None:
        weights = [1.0 / len(scenarios)] * len(scenarios)

    combined_shocks = {}

    for scenario, weight in zip(scenarios, weights):
        for factor, shock in scenario.shocks.items():
            if factor in combined_shocks:
                combined_shocks[factor] += shock * weight
            else:
                combined_shocks[factor] = shock * weight

    return Scenario(
        name=name,
        description=f"Combined from {len(scenarios)} scenarios",
        scenario_type=ScenarioType.CUSTOM,
        shocks=combined_shocks,
    )
