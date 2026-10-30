"""
Regulatory Stress Testing Frameworks

Implements supervisory stress testing:
- CCAR (Comprehensive Capital Analysis and Review) - US Federal Reserve
- DFAST (Dodd-Frank Act Stress Testing) - US
- EBA Stress Testing - European Banking Authority

Provides scenario generation and capital projection under stress.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class ScenarioSeverity(Enum):
    """Stress scenario severity levels."""
    BASELINE = "baseline"
    ADVERSE = "adverse"
    SEVERELY_ADVERSE = "severely_adverse"


@dataclass
class StressPeriod:
    """
    Time period within stress scenario.

    Attributes:
        quarter: Quarter number (1-9 for typical CCAR)
        gdp_growth: GDP growth rate (%)
        unemployment: Unemployment rate (%)
        interest_rate_short: Short-term interest rate change (bps)
        interest_rate_long: Long-term interest rate change (bps)
        equity_market: Equity market return (%)
        house_prices: House price change (%)
        commercial_real_estate: CRE price change (%)
        vix: Market volatility index level
    """
    quarter: int
    gdp_growth: float
    unemployment: float
    interest_rate_short: float
    interest_rate_long: float
    equity_market: float
    house_prices: float = 0.0
    commercial_real_estate: float = 0.0
    vix: float = 20.0


@dataclass
class StressScenario:
    """
    Complete stress scenario definition.

    Multi-period scenario with macroeconomic shocks.
    """
    name: str
    severity: ScenarioSeverity
    periods: List[StressPeriod]
    description: str = ""

    @property
    def num_quarters(self) -> int:
        """Number of quarters in scenario."""
        return len(self.periods)

    def cumulative_gdp_impact(self) -> float:
        """Cumulative GDP growth over scenario."""
        return sum(p.gdp_growth for p in self.periods)


@dataclass
class CapitalProjection:
    """
    Capital projection under stress scenario.

    Attributes:
        quarters: Time periods
        cet1_capital: CET1 capital levels
        tier1_capital: Tier 1 capital levels
        total_capital: Total capital levels
        rwa: Risk-weighted assets
        cet1_ratios: CET1 ratio over time
        tier1_ratios: Tier 1 ratios
        total_capital_ratios: Total capital ratios
        minimum_cet1: Minimum CET1 ratio reached
    """
    quarters: List[int]
    cet1_capital: List[float]
    tier1_capital: List[float]
    total_capital: List[float]
    rwa: List[float]
    cet1_ratios: List[float]
    tier1_ratios: List[float]
    total_capital_ratios: List[float]

    @property
    def minimum_cet1(self) -> float:
        """Minimum CET1 ratio over projection period."""
        return min(self.cet1_ratios) if self.cet1_ratios else 0.0

    def passes_stress(self, min_cet1: float = 4.5) -> bool:
        """Check if capital remains above minimum under stress."""
        return self.minimum_cet1 >= min_cet1


@dataclass
class StressTestResult:
    """
    Complete stress test result.

    Attributes:
        scenario: Scenario definition
        projections: Capital projections
        pre_provision_net_revenue: PPNR over scenario
        credit_losses: Loan losses over scenario
        trading_losses: Mark-to-market losses
        passes: Whether bank passes stress test
    """
    scenario: StressScenario
    projections: CapitalProjection
    pre_provision_net_revenue: List[float]
    credit_losses: List[float]
    trading_losses: List[float]
    passes: bool


class SeverelyAdverseScenario:
    """
    Severely Adverse scenario generator.

    Based on historical severe recessions (2008 GFC-like scenario).
    """

    @staticmethod
    def generate(num_quarters: int = 9) -> StressScenario:
        """
        Generate severely adverse scenario.

        Typical assumptions:
        - Deep recession with -3% to -5% GDP
        - Unemployment rises 5+ percentage points
        - 50%+ equity market decline
        - Interest rates fall sharply initially
        """
        periods = []

        for q in range(1, num_quarters + 1):
            # GDP declines sharply in first 4 quarters
            if q <= 4:
                gdp = -3.0 - (q * 0.5)
            else:
                gdp = -1.0 + ((q - 4) * 0.5)  # Recovery starts

            # Unemployment rises to peak around Q6
            unemployment = min(10.0 + q, 15.0)

            # Equity markets crash
            if q == 1:
                equity = -40.0
            elif q <= 3:
                equity = -20.0
            else:
                equity = 5.0  # Modest recovery

            # Interest rates
            if q <= 2:
                ir_short = -200  # Flight to quality
                ir_long = -100
            else:
                ir_short = -50
                ir_long = 0

            # House prices fall
            house_price = -5.0 if q <= 6 else -2.0

            # Volatility spikes
            vix = 60.0 if q <= 3 else 40.0

            periods.append(StressPeriod(
                quarter=q,
                gdp_growth=gdp,
                unemployment=unemployment,
                interest_rate_short=ir_short,
                interest_rate_long=ir_long,
                equity_market=equity,
                house_prices=house_price,
                commercial_real_estate=-8.0 if q <= 6 else -3.0,
                vix=vix
            ))

        return StressScenario(
            name="Severely Adverse",
            severity=ScenarioSeverity.SEVERELY_ADVERSE,
            periods=periods,
            description="Deep recession scenario similar to 2008 financial crisis"
        )


class AdverseScenario:
    """Adverse scenario (moderate recession)."""

    @staticmethod
    def generate(num_quarters: int = 9) -> StressScenario:
        """Generate adverse scenario (moderate stress)."""
        periods = []

        for q in range(1, num_quarters + 1):
            gdp = -1.0 if q <= 4 else 0.5
            unemployment = min(7.0 + q * 0.3, 9.0)
            equity = -15.0 if q <= 2 else 0.0
            ir_short = -100 if q <= 2 else 0
            ir_long = -50

            periods.append(StressPeriod(
                quarter=q,
                gdp_growth=gdp,
                unemployment=unemployment,
                interest_rate_short=ir_short,
                interest_rate_long=ir_long,
                equity_market=equity,
                house_prices=-2.0,
                commercial_real_estate=-3.0,
                vix=35.0 if q <= 3 else 25.0
            ))

        return StressScenario(
            name="Adverse",
            severity=ScenarioSeverity.ADVERSE,
            periods=periods,
            description="Moderate recession scenario"
        )


class BaselineScenario:
    """Baseline scenario (most likely path)."""

    @staticmethod
    def generate(num_quarters: int = 9) -> StressScenario:
        """Generate baseline scenario."""
        periods = []

        for q in range(1, num_quarters + 1):
            periods.append(StressPeriod(
                quarter=q,
                gdp_growth=2.0,
                unemployment=4.5,
                interest_rate_short=0,
                interest_rate_long=0,
                equity_market=8.0,
                house_prices=3.0,
                commercial_real_estate=2.0,
                vix=15.0
            ))

        return StressScenario(
            name="Baseline",
            severity=ScenarioSeverity.BASELINE,
            periods=periods,
            description="Most likely economic path"
        )


class StressTestFramework:
    """
    Generic stress testing framework.

    Projects capital under stressed scenarios.
    """

    def __init__(
        self,
        initial_cet1: float,
        initial_rwa: float,
        scenario: StressScenario
    ) -> None:
        """
        Initialize stress test.

        Args:
            initial_cet1: Starting CET1 capital
            initial_rwa: Starting RWA
            scenario: Stress scenario to apply
        """
        self.initial_cet1 = initial_cet1
        self.initial_rwa = initial_rwa
        self.scenario = scenario

    def project_capital(
        self,
        ppnr_quarterly: List[float],
        credit_loss_rates: List[float],
        rwa_growth_rates: List[float]
    ) -> CapitalProjection:
        """
        Project capital ratios under stress.

        Args:
            ppnr_quarterly: Pre-provision net revenue each quarter
            credit_loss_rates: Credit loss rates (% of loans)
            rwa_growth_rates: RWA growth/decline rates

        Returns:
            CapitalProjection with complete trajectory
        """
        quarters = []
        cet1_levels = [self.initial_cet1]
        rwa_levels = [self.initial_rwa]
        cet1_ratios = []

        for i, period in enumerate(self.scenario.periods):
            # Capital change = PPNR - Losses - Dividends
            ppnr = ppnr_quarterly[i] if i < len(ppnr_quarterly) else 0
            loss_rate = credit_loss_rates[i] if i < len(credit_loss_rates) else 0
            credit_loss = rwa_levels[-1] * loss_rate

            # Simplified: no dividends during stress
            capital_change = ppnr - credit_loss

            # New CET1
            new_cet1 = cet1_levels[-1] + capital_change
            cet1_levels.append(new_cet1)

            # RWA evolution
            rwa_growth = rwa_growth_rates[i] if i < len(rwa_growth_rates) else 0
            new_rwa = rwa_levels[-1] * (1 + rwa_growth)
            rwa_levels.append(new_rwa)

            # Ratio
            cet1_ratio = (new_cet1 / new_rwa) * 100 if new_rwa > 0 else 0
            cet1_ratios.append(cet1_ratio)

            quarters.append(period.quarter)

        return CapitalProjection(
            quarters=quarters,
            cet1_capital=cet1_levels[1:],
            tier1_capital=cet1_levels[1:],  # Simplified
            total_capital=cet1_levels[1:],
            rwa=rwa_levels[1:],
            cet1_ratios=cet1_ratios,
            tier1_ratios=cet1_ratios,  # Simplified
            total_capital_ratios=cet1_ratios
        )


class CCARTest(StressTestFramework):
    """
    CCAR (Comprehensive Capital Analysis and Review) stress test.

    Federal Reserve's annual stress test for large US banks.
    """

    def run(
        self,
        ppnr_model: Optional[List[float]] = None,
        loss_rates_model: Optional[List[float]] = None
    ) -> StressTestResult:
        """
        Run CCAR stress test.

        Returns:
            Complete stress test results
        """
        # Default models if not provided
        if ppnr_model is None:
            ppnr_model = [100_000] * self.scenario.num_quarters

        if loss_rates_model is None:
            # Higher losses in severe stress
            base_rate = 0.02 if self.scenario.severity == ScenarioSeverity.SEVERELY_ADVERSE else 0.01
            loss_rates_model = [base_rate * (1 + 0.1 * q) for q in range(self.scenario.num_quarters)]

        rwa_growth = [0.0] * self.scenario.num_quarters  # Assume stable RWA

        projections = self.project_capital(ppnr_model, loss_rates_model, rwa_growth)

        # Calculate losses
        credit_losses = [self.initial_rwa * rate for rate in loss_rates_model]

        # Test passes if min CET1 > 4.5%
        passes = projections.passes_stress(min_cet1=4.5)

        return StressTestResult(
            scenario=self.scenario,
            projections=projections,
            pre_provision_net_revenue=ppnr_model,
            credit_losses=credit_losses,
            trading_losses=[0.0] * self.scenario.num_quarters,
            passes=passes
        )


class DFASTTest(StressTestFramework):
    """
    DFAST (Dodd-Frank Act Stress Testing).

    Similar to CCAR but mandated by Dodd-Frank Act.
    """

    def run(self) -> StressTestResult:
        """Run DFAST stress test."""
        # Simplified implementation
        ccar = CCARTest(self.initial_cet1, self.initial_rwa, self.scenario)
        return ccar.run()


def project_capital_ratios(
    initial_capital: float,
    initial_rwa: float,
    earnings: List[float],
    losses: List[float],
    rwa_changes: List[float]
) -> List[float]:
    """
    Project capital ratios over time.

    Args:
        initial_capital: Starting capital
        initial_rwa: Starting RWA
        earnings: Net income each period
        losses: Credit/market losses each period
        rwa_changes: RWA changes each period

    Returns:
        List of capital ratios for each period
    """
    capital = initial_capital
    rwa = initial_rwa
    ratios = []

    n = min(len(earnings), len(losses), len(rwa_changes))
    for i in range(n):
        # Update capital
        capital += earnings[i] - losses[i]

        # Update RWA
        rwa += rwa_changes[i]

        # Calculate ratio
        ratio = (capital / rwa) * 100 if rwa > 0 else 0
        ratios.append(ratio)

    return ratios
