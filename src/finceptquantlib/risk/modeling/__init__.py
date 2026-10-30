"""
Risk Modeling

XVA, tail risk, copulas, extreme value theory.
"""

from .correlation_stress import (
    CorrelationScenario,
    CorrelationStressTester,
)

from .tail_risk import (
    CoherentRiskMeasure,
    TailRiskAnalyzer,
    TailRiskMetrics,
    conditional_value_at_risk,
    expected_shortfall_optimization,
    tail_conditional_expectation,
    tail_dependence_coefficient,
    worst_conditional_expectation,
)

from .xva import (
    CVACalculator,
    CreditParameters,
    FVACalculator,
    FundingParameters,
    KVACalculator,
    MVACalculator,
    XVAEngine,
    XVAResult,
    calculate_expected_positive_exposure,
    calculate_potential_future_exposure,
    simulate_exposure_profile,
)

from .copulas import *
from .evt import *

__all__ = [
    "CVACalculator",
    "CoherentRiskMeasure",
    "CorrelationScenario",
    "CorrelationStressTester",
    "CreditParameters",
    "FVACalculator",
    "FundingParameters",
    "KVACalculator",
    "MVACalculator",
    "TailRiskAnalyzer",
    "TailRiskMetrics",
    "XVAEngine",
    "XVAResult",
    "calculate_expected_positive_exposure",
    "calculate_potential_future_exposure",
    "conditional_value_at_risk",
    "expected_shortfall_optimization",
    "simulate_exposure_profile",
    "tail_conditional_expectation",
    "tail_dependence_coefficient",
    "worst_conditional_expectation",
]

# Extend __all__ with copulas and evt
from .copulas import __all__ as _copulas_all
from .evt import __all__ as _evt_all
__all__.extend(_copulas_all)
__all__.extend(_evt_all)
