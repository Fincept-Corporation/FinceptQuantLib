"""
FinceptQuantLib - quantitative finance library by Fincept Corporation.

Pricing, risk and analytics for fixed income, derivatives and portfolios,
implemented in Python on top of SciPy.

Subpackages:
    core         Automatic differentiation, financial types, legs and periods
    scheduling   Day counts, holiday calendars, business-day rules, schedules
    curves       Yield, inflation and parametric curves; curve operations
    instruments  Bonds, swaps, FRAs, futures, options, FX, credit, exotics
    pricing      Closed-form models (Black-Scholes, Black-76, Bachelier) and trees
    models       Heston, SABR, jump-diffusion, local vol and short-rate models
    volatility   Volatility surfaces, SABR calibration, local volatility
    solver       Curve bootstrapping, root finding and fixed-income analytics
    risk         VaR/ES, sensitivities, stress testing, XVA, copulas, EVT
    portfolio    Mean-variance, Black-Litterman, risk parity
    regulatory   Basel III/IV, SA-CCR, FRTB, IFRS 9, LCR/NSFR
    analysis     Financial statement analysis and valuation (DCF, residual income)
    stochastic   Stochastic processes and SDE simulation
    statistics   Probability distributions and time-series models
    numerical    Linear algebra, interpolation, integration, optimisation, FFT
    ml           Credit scoring, model validation, ML for curves and portfolios
    economics    Utility theory, game theory, general equilibrium
    physics      Information theory, statistical mechanics, thermodynamics

Every public name of these subpackages is also available at the top level.
When two subpackages export the same name, the top-level binding follows
this precedence (highest first): core, scheduling, curves, pricing, solver,
portfolio, risk, instruments, volatility, models, analysis, regulatory,
stochastic, statistics, economics, physics, ml. Import from the subpackage
to get the other variant, e.g. ``finceptquantlib.ml.CalibrationResult``.
"""

__version__ = "1.0.0"
__author__ = "Fincept Corporation"
__license__ = "MIT"

from finceptquantlib import (
    analysis,
    core,
    curves,
    economics,
    instruments,
    ml,
    models,
    numerical,
    physics,
    portfolio,
    pricing,
    regulatory,
    risk,
    scheduling,
    solver,
    statistics,
    stochastic,
    volatility,
)

# Star imports run from lowest to highest precedence: later imports win.
from finceptquantlib.ml import *
from finceptquantlib.physics import *
from finceptquantlib.economics import *
from finceptquantlib.statistics import *
from finceptquantlib.stochastic import *
from finceptquantlib.regulatory import *
from finceptquantlib.analysis import *
from finceptquantlib.models import *
from finceptquantlib.volatility import *
from finceptquantlib.instruments import *
from finceptquantlib.risk import *
from finceptquantlib.portfolio import *
from finceptquantlib.solver import *
from finceptquantlib.pricing import *
from finceptquantlib.curves import *
from finceptquantlib.scheduling import *
from finceptquantlib.core import *

# Aliases that disambiguate names shared between subpackages
from finceptquantlib.core import FloatLeg as CoreFloatLeg
from finceptquantlib.stochastic import (
    BatesModel as StochasticBates,
    KouProcess as StochasticKou,
    MertonProcess as StochasticMerton,
    VarianceGammaProcess as StochasticVG,
)

_ALIASES = [
    "CoreFloatLeg",
    "StochasticBates",
    "StochasticKou",
    "StochasticMerton",
    "StochasticVG",
]

__all__ = ["__version__"]
for _module in (
    core, scheduling, curves, pricing, solver, portfolio, risk, instruments,
    volatility, models, analysis, regulatory, stochastic, statistics,
    economics, physics, ml,
):
    __all__.extend(_module.__all__)
__all__.extend(_ALIASES)
__all__ = list(dict.fromkeys(__all__))

del _module, _ALIASES
