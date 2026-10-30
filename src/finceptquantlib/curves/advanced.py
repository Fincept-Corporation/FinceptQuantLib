"""
Advanced Curve Features

Enhanced curve functionality:
- Constrained curve fitting (monotonicity, smoothness)
- Real-time incremental updates
- Curve caching and versioning
- Inflation seasonality
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Dict
from copy import deepcopy

from scipy.optimize import minimize

from finceptquantlib.curves.curve import Curve, ZeroCurve

# Rates are ~1e-2, so squared errors are ~1e-6 and would sit inside the optimiser's
# default tolerances; working in basis points keeps the fit meaningful.
_BP = 1e4


@dataclass
class CurveConstraints:
    """
    Constraints for curve fitting.

    Attributes:
        monotonic: Enforce monotonicity (forward rates non-decreasing)
        smooth: Enforce smoothness (limit curvature)
        positive: Enforce positive rates
        max_curvature: Maximum allowed second derivative
        min_rate: Minimum rate
        max_rate: Maximum rate
    """
    monotonic: bool = False
    smooth: bool = False
    positive: bool = True
    max_curvature: Optional[float] = None
    min_rate: Optional[float] = None
    max_rate: Optional[float] = None


def _discrete_forwards(rates: List[float], tenors: List[float]) -> List[float]:
    """Forward rate over each tenor interval implied by continuously compounded zero rates."""
    forwards = [rates[0]]
    for i in range(1, len(rates)):
        forwards.append(
            (rates[i] * tenors[i] - rates[i - 1] * tenors[i - 1]) / (tenors[i] - tenors[i - 1])
        )
    return forwards


def _second_derivatives(rates: List[float], tenors: List[float]) -> List[float]:
    """Finite-difference second derivative of the rates at each interior tenor."""
    result = []
    for i in range(1, len(rates) - 1):
        dt1 = tenors[i] - tenors[i - 1]
        dt2 = tenors[i + 1] - tenors[i]
        if dt1 > 0 and dt2 > 0:
            first_diff_left = (rates[i] - rates[i - 1]) / dt1
            first_diff_right = (rates[i + 1] - rates[i]) / dt2
            result.append((first_diff_right - first_diff_left) / ((dt1 + dt2) / 2))
    return result


class ConstrainedCurveFitter:
    """
    Fit a zero curve to market zero rates subject to constraints.

    Minimises the squared fitting error (plus a curvature penalty when
    ``smooth`` is set) over the zero rates at the market tenors, subject to:
    - Positivity and [min_rate, max_rate] bounds
    - Monotonicity (forward rates non-decreasing)
    - Maximum curvature (|second derivative| <= max_curvature)

    Examples:
        >>> fitter = ConstrainedCurveFitter(
        ...     constraints=CurveConstraints(monotonic=True, positive=True)
        ... )
        >>> curve = fitter.fit(date(2024, 1, 2), [1, 2, 5, 10], [0.040, 0.042, 0.041, 0.045])
    """

    def __init__(
        self,
        constraints: Optional[CurveConstraints] = None,
        smoothness_weight: float = 1.0,
    ) -> None:
        """
        Initialize fitter.

        Args:
            constraints: Curve constraints
            smoothness_weight: Weight of the curvature penalty when ``smooth`` is set
        """
        self.constraints = constraints or CurveConstraints()
        self.smoothness_weight = smoothness_weight

    def fit(self, reference_date: date, tenors: List[float], rates: List[float]) -> ZeroCurve:
        """
        Fit curve with constraints.

        Args:
            reference_date: Curve valuation date
            tenors: Market tenors in years (positive, strictly increasing)
            rates: Market zero rates (continuously compounded, decimals)

        Returns:
            Fitted zero curve at the market tenors

        Raises:
            ValueError: If the inputs or constraints are inconsistent, or the fit fails
        """
        n = len(tenors)
        if n < 2 or len(rates) != n:
            raise ValueError("tenors and rates must have the same length, with at least 2 points")
        if tenors[0] <= 0 or any(b <= a for a, b in zip(tenors, tenors[1:])):
            raise ValueError("tenors must be positive and strictly increasing")

        c = self.constraints
        lower = c.min_rate
        if c.positive:
            lower = 0.0 if lower is None else max(lower, 0.0)
        upper = c.max_rate
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("min_rate (or 0 when positive) must not exceed max_rate")

        smoothness = SmoothnessConstraint(self.smoothness_weight)

        # Variables are zero rates in basis points.
        def objective(x_bp: List[float]) -> float:
            x = [v / _BP for v in x_bp]
            error = sum((xi - ri) ** 2 for xi, ri in zip(x, rates))
            if c.smooth:
                error += smoothness.penalty(x, tenors)
            return error * _BP * _BP

        constraints = []
        if c.monotonic:
            def forward_increments(x_bp: List[float]) -> List[float]:
                fwd = _discrete_forwards(list(x_bp), tenors)
                return [fwd[i + 1] - fwd[i] for i in range(n - 1)]
            constraints.append({"type": "ineq", "fun": forward_increments})
        if c.max_curvature is not None and n > 2:
            limit_bp = c.max_curvature * _BP

            def curvature_room(x_bp: List[float]) -> List[float]:
                curvature = _second_derivatives(list(x_bp), tenors)
                return [limit_bp - v for v in curvature] + [limit_bp + v for v in curvature]
            constraints.append({"type": "ineq", "fun": curvature_room})

        bounds = [
            (None if lower is None else lower * _BP, None if upper is None else upper * _BP)
        ] * n
        x0 = []
        for r in rates:
            if lower is not None:
                r = max(r, lower)
            if upper is not None:
                r = min(r, upper)
            x0.append(r * _BP)

        if constraints:
            result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        else:
            result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        if not result.success:
            raise ValueError(f"Constrained curve fit failed: {result.message}")

        return ZeroCurve(reference_date, list(tenors), [float(v) / _BP for v in result.x])


class IncrementalCurveUpdater:
    """
    Real-time incremental curve updates.

    Efficiently updates curves when subset of instruments change,
    without full re-bootstrap.

    Examples:
        >>> updater = IncrementalCurveUpdater(base_curve)  # doctest: +SKIP
        >>> updated = updater.update_instrument(index=5, new_price=99.5)  # doctest: +SKIP
    """

    def __init__(self, base_curve: Curve) -> None:
        """
        Initialize updater.

        Args:
            base_curve: Base curve to update
        """
        self.base_curve = base_curve
        self.curve_history: List[Curve] = [deepcopy(base_curve)]

    def update_instrument(
        self,
        index: int,
        new_price: float,
        instrument_sensitivity: Optional[float] = None
    ) -> Curve:
        """
        Update curve for single instrument price change.

        Args:
            index: Instrument index in original bootstrap
            new_price: New market price
            instrument_sensitivity: Price sensitivity to rate change

        Returns:
            Updated curve
        """
        # Simplified: return base curve
        # Full version would use sensitivity-based update
        updated = deepcopy(self.base_curve)
        self.curve_history.append(updated)
        return updated

    def rollback(self, steps: int = 1) -> Curve:
        """Roll back to previous curve version."""
        if len(self.curve_history) <= steps:
            return self.curve_history[0]
        return self.curve_history[-(steps + 1)]


@dataclass
class CurveVersion:
    """Curve version for tracking."""
    curve: Curve
    timestamp: date
    version_id: int
    build_time_ms: float = 0.0


class CurveDatabase:
    """
    Historical curve storage and retrieval.

    Stores historical curves for back-testing and analysis.

    Examples:
        >>> db = CurveDatabase()
        >>> db.store_curve("USD-OIS", curve, date(2024, 1, 15))  # doctest: +SKIP
        >>> historical = db.get_curve("USD-OIS", date(2024, 1, 15))
    """

    def __init__(self) -> None:
        """Initialize curve database."""
        self.curves: Dict[str, List[CurveVersion]] = {}
        self.version_counter = 0

    def store_curve(
        self,
        curve_name: str,
        curve: Curve,
        as_of_date: date,
        build_time_ms: float = 0.0
    ) -> int:
        """
        Store curve with versioning.

        Args:
            curve_name: Curve identifier
            curve: Curve object
            as_of_date: Curve date
            build_time_ms: Build time in milliseconds

        Returns:
            Version ID
        """
        if curve_name not in self.curves:
            self.curves[curve_name] = []

        version = CurveVersion(
            curve=deepcopy(curve),
            timestamp=as_of_date,
            version_id=self.version_counter,
            build_time_ms=build_time_ms
        )

        self.curves[curve_name].append(version)
        self.version_counter += 1

        return version.version_id

    def get_curve(
        self,
        curve_name: str,
        as_of_date: date,
        exact_match: bool = False
    ) -> Optional[Curve]:
        """
        Retrieve curve for specific date.

        Args:
            curve_name: Curve identifier
            as_of_date: Date to retrieve
            exact_match: If True, must match exactly; else use latest before date

        Returns:
            Curve or None if not found
        """
        if curve_name not in self.curves:
            return None

        versions = self.curves[curve_name]

        if exact_match:
            # Find exact date
            for v in versions:
                if v.timestamp == as_of_date:
                    return v.curve
            return None
        else:
            # Find latest on or before date
            valid = [v for v in versions if v.timestamp <= as_of_date]
            if not valid:
                return None
            latest = max(valid, key=lambda v: v.timestamp)
            return latest.curve

    def get_curve_history(
        self,
        curve_name: str,
        start_date: date,
        end_date: date
    ) -> List[CurveVersion]:
        """Get all curve versions in date range."""
        if curve_name not in self.curves:
            return []

        return [v for v in self.curves[curve_name]
                if start_date <= v.timestamp <= end_date]


class InflationSeasonality:
    """
    Inflation Seasonality Adjustments.

    Accounts for seasonal patterns in inflation (e.g., energy prices in winter).

    Examples:
        >>> seasonality = InflationSeasonality()
        >>> seasonality.set_monthly_factors([1.02, 1.01, 1.0, 0.99] * 3)  # 12 factors
        >>> round(seasonality.adjust(base_value=250.0, month=12), 4)  # December
        246.2687
    """

    def __init__(self) -> None:
        """Initialize seasonality."""
        # Default: no seasonality (all factors = 1.0)
        self.monthly_factors = [1.0] * 12

    def set_monthly_factors(self, factors: List[float]) -> None:
        """
        Set seasonal adjustment factors.

        Args:
            factors: 12 monthly factors (Jan, Feb, ..., Dec)
                    >1.0 = higher inflation that month
                    <1.0 = lower inflation that month
        """
        if len(factors) != 12:
            raise ValueError("Must provide exactly 12 monthly factors")

        # Normalize so average = 1.0
        avg = sum(factors) / 12
        self.monthly_factors = [f / avg for f in factors]

    def adjust(self, base_value: float, month: int) -> float:
        """
        Apply seasonal adjustment.

        Args:
            base_value: Base inflation index value
            month: Month (1-12)

        Returns:
            Seasonally adjusted value
        """
        if not (1 <= month <= 12):
            raise ValueError(f"Month must be 1-12, got {month}")

        factor = self.monthly_factors[month - 1]
        return base_value * factor

    def deseasonalize(self, observed_value: float, month: int) -> float:
        """Remove seasonal component."""
        if not (1 <= month <= 12):
            raise ValueError(f"Month must be 1-12, got {month}")

        factor = self.monthly_factors[month - 1]
        return observed_value / factor


class SmoothnessConstraint:
    """
    Enforces smoothness in curve fitting.

    Penalizes large second derivatives (curvature).
    """

    def __init__(self, penalty_weight: float = 1.0) -> None:
        """
        Initialize smoothness constraint.

        Args:
            penalty_weight: Weight for smoothness penalty
        """
        self.penalty_weight = penalty_weight

    def penalty(self, rates: List[float], tenors: List[float]) -> float:
        """
        Calculate smoothness penalty.

        Penalty = sum of (second differences)²

        Args:
            rates: Rate values
            tenors: Tenor points

        Returns:
            Penalty value
        """
        total_penalty = sum(d ** 2 for d in _second_derivatives(rates, tenors))
        return self.penalty_weight * total_penalty


class MonotonicityConstraint:
    """
    Enforces monotonicity in forward rates.

    Forward rates should be non-decreasing (no negative slope).
    """

    def __init__(self) -> None:
        """Initialize monotonicity constraint."""

    def check_monotonic(self, forward_rates: List[float]) -> bool:
        """
        Check if forward rates are monotonic.

        Args:
            forward_rates: List of forward rates

        Returns:
            True if monotonic
        """
        for i in range(len(forward_rates) - 1):
            if forward_rates[i+1] < forward_rates[i]:
                return False
        return True

    def enforce_monotonic(self, forward_rates: List[float]) -> List[float]:
        """
        Enforce monotonicity by adjusting rates.

        Args:
            forward_rates: Original rates

        Returns:
            Monotonic rates
        """
        adjusted = forward_rates[:]

        for i in range(1, len(adjusted)):
            if adjusted[i] < adjusted[i-1]:
                adjusted[i] = adjusted[i-1]  # Floor at previous rate

        return adjusted
