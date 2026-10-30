"""
Solver Module

Numerical solvers for finance:
- 1D root finding (brent, newton, bisect, secant, ridder)
- Multi-dim root finding (NewtonRaphson, Broyden)
- Nonlinear least squares (LevenbergMarquardt, BoundedLevenbergMarquardt, GaussNewton)
- Finance-specific (irr, bond_yield, implied_vol, z_spread, oas, convexity_adjustment)
- Curve bootstrapping (CurveBootstrapper, GlobalCurveFitter)
- Model calibration (calibrate_hull_white, calibrate_vasicek)
"""

# 1D Root Finding
from finceptquantlib.solver.root1d import (
    brent,
    newton,
    bisect,
    secant,
    ridder,
    RootResult,
    find_root,
    find_root_newton,
)

# Multi-dimensional Root Finding
from finceptquantlib.solver.rootnd import (
    NewtonRaphson,
    Broyden,
    RootResultND,
    find_root_nd,
)

# Nonlinear Least Squares
from finceptquantlib.solver.leastsq import (
    LevenbergMarquardt,
    BoundedLevenbergMarquardt,
    GaussNewton,
    LeastSquaresResult,
    least_squares,
)

# Linear Algebra
from finceptquantlib.solver.linear import (
    solve,
    lstsq,
    lu_decompose,
    dot,
    norm,
    matvec,
    transpose,
)

# Jacobian Computation
from finceptquantlib.solver.jacobian import (
    finite_difference_jacobian,
    finite_difference_gradient,
    finite_difference_derivative,
    finite_difference_hessian,
    JacobianComputer,
    parallel_jacobian,
    parallel_gradient,
    ParallelJacobianComputer,
)

# Finance-Specific Solvers
from finceptquantlib.solver.finance import (
    # IRR / Yield
    irr,
    xirr,
    bond_yield,
    # Implied Volatility
    implied_vol,
    implied_vol_black76,
    # Rates
    par_rate,
    forward_rate,
    zero_rate,
    discount_factor,
    # Duration / Convexity
    duration_from_yield,
    modified_duration,
    convexity,
    # Spread Solvers
    z_spread,
    oas,
    asw_spread,
    i_spread,
    g_spread,
    # Convexity Adjustment
    convexity_adjustment,
    futures_to_forward,
    forward_to_futures,
    # Model Calibration
    HullWhiteParams,
    CalibrationResult,
    calibrate_hull_white,
    calibrate_vasicek,
    # Utilities
    implied_repo_rate,
    basis,
    carry,
    dv01,
    pv01,
)

# Curve Bootstrapping
from finceptquantlib.solver.bootstrap import (
    CurveBootstrapper,
    GlobalCurveFitter,
    BootstrapConfig,
    CalibrationResult as BootstrapCalibrationResult,
    CalibrationInstrument,
    InstrumentType,
    bootstrap_curve,
)

__all__ = [
    # 1D Root Finding
    "brent",
    "newton",
    "bisect",
    "secant",
    "ridder",
    "RootResult",
    "find_root",
    "find_root_newton",
    # Multi-dim Root Finding
    "NewtonRaphson",
    "Broyden",
    "RootResultND",
    "find_root_nd",
    # Least Squares
    "LevenbergMarquardt",
    "BoundedLevenbergMarquardt",
    "GaussNewton",
    "LeastSquaresResult",
    "least_squares",
    # Linear Algebra
    "solve",
    "lstsq",
    "lu_decompose",
    "dot",
    "norm",
    "matvec",
    "transpose",
    # Jacobian
    "finite_difference_jacobian",
    "finite_difference_gradient",
    "finite_difference_derivative",
    "finite_difference_hessian",
    "JacobianComputer",
    "parallel_jacobian",
    "parallel_gradient",
    "ParallelJacobianComputer",
    # Finance - IRR/Yield
    "irr",
    "xirr",
    "bond_yield",
    # Finance - Implied Vol
    "implied_vol",
    "implied_vol_black76",
    # Finance - Rates
    "par_rate",
    "forward_rate",
    "zero_rate",
    "discount_factor",
    # Finance - Duration/Convexity
    "duration_from_yield",
    "modified_duration",
    "convexity",
    # Finance - Spreads
    "z_spread",
    "oas",
    "asw_spread",
    "i_spread",
    "g_spread",
    # Finance - Convexity Adjustment
    "convexity_adjustment",
    "futures_to_forward",
    "forward_to_futures",
    # Finance - Model Calibration
    "HullWhiteParams",
    "CalibrationResult",
    "calibrate_hull_white",
    "calibrate_vasicek",
    # Finance - Utilities
    "implied_repo_rate",
    "basis",
    "carry",
    "dv01",
    "pv01",
    # Bootstrapping
    "CurveBootstrapper",
    "GlobalCurveFitter",
    "BootstrapConfig",
    "BootstrapCalibrationResult",
    "CalibrationInstrument",
    "InstrumentType",
    "bootstrap_curve",
]
