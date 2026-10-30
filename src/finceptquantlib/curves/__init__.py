"""
Yield Curve Module

Provides yield curve construction, interpolation, and analytics:
- Multiple curve types (discount, zero, forward)
- Various interpolation methods
- Curve operations (shift, twist, butterfly, roll)
- Parametric models (Nelson-Siegel, Svensson)
- Inflation curves
"""

from finceptquantlib.curves.curve import (
    Curve,
    DiscountCurve,
    ZeroCurve,
    ForwardCurve,
)

from finceptquantlib.curves.interpolation import (
    Interpolator,
    LinearInterpolator,
    LogLinearInterpolator,
    CubicSplineInterpolator,
    MonotoneConvexInterpolator,
    FlatForwardInterpolator,
    FlatBackwardInterpolator,
)

from finceptquantlib.curves.parametric import (
    NelsonSiegelCurve,
    NelsonSiegelSvenssonCurve,
)

from finceptquantlib.curves.inflation import (
    InflationCurve,
    RealRateCurve,
    SeasonalityModel,
    bootstrap_inflation_curve,
    CPIObservation,
)

from finceptquantlib.curves.operations import (
    # Base wrapper
    CurveWrapper,
    # Shift types
    ShiftType,
    ShiftSpace,
    # Curve wrappers
    ShiftedCurve,
    TwistedCurve,
    ButterflyShiftedCurve,
    TimeShiftedCurve,
    ScaledCurve,
    ProxyCurve,
    CompositeCurve,
    KeyRateShiftedCurve,
    # Convenience functions
    parallel_shift,
    twist_curve,
    butterfly_shift,
    roll_curve,
    key_rate_shift,
)

from finceptquantlib.curves.multicurve import (
    MultiCurveFramework,
    CurveSet,
    OISDiscountCurve,
    IBORProjectionCurve,
    CrossCurrencyBasisCurve,
    calibrate_multicurve,
)

from finceptquantlib.curves.advanced import (
    ConstrainedCurveFitter,
    CurveConstraints,
    IncrementalCurveUpdater,
    CurveDatabase,
    CurveVersion,
    InflationSeasonality,
    SmoothnessConstraint,
    MonotonicityConstraint,
)

__all__ = [
    # Curves
    "Curve",
    "DiscountCurve",
    "ZeroCurve",
    "ForwardCurve",
    # Interpolators
    "Interpolator",
    "LinearInterpolator",
    "LogLinearInterpolator",
    "CubicSplineInterpolator",
    "MonotoneConvexInterpolator",
    "FlatForwardInterpolator",
    "FlatBackwardInterpolator",
    # Parametric
    "NelsonSiegelCurve",
    "NelsonSiegelSvenssonCurve",
    # Inflation
    "InflationCurve",
    "RealRateCurve",
    "SeasonalityModel",
    "bootstrap_inflation_curve",
    "CPIObservation",
    # Curve Operations
    "CurveWrapper",
    "ShiftType",
    "ShiftSpace",
    "ShiftedCurve",
    "TwistedCurve",
    "ButterflyShiftedCurve",
    "TimeShiftedCurve",
    "ScaledCurve",
    "ProxyCurve",
    "CompositeCurve",
    "KeyRateShiftedCurve",
    # Convenience functions
    "parallel_shift",
    "twist_curve",
    "butterfly_shift",
    "roll_curve",
    "key_rate_shift",
    # Multi-Curve Framework
    "MultiCurveFramework",
    "CurveSet",
    "OISDiscountCurve",
    "IBORProjectionCurve",
    "CrossCurrencyBasisCurve",
    "calibrate_multicurve",
    # Advanced Features
    "ConstrainedCurveFitter",
    "CurveConstraints",
    "IncrementalCurveUpdater",
    "CurveDatabase",
    "CurveVersion",
    "InflationSeasonality",
    "SmoothnessConstraint",
    "MonotonicityConstraint",
]
