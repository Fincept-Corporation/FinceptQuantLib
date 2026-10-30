"""
Yield Curve Classes

Implements yield curve representations:
- DiscountCurve: Discount factors as a function of time
- ZeroCurve: Zero (spot) rates as a function of maturity
- ForwardCurve: Forward rates

Curves support:
- Multiple interpolation methods
- Automatic differentiation for sensitivities
- Curve operations (shift, twist, roll)
- Conversion between curve types
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, timedelta
from typing import Union, List
import math

from finceptquantlib.curves.interpolation import (
    Interpolator,
    LinearInterpolator,
    LogLinearInterpolator,
    CubicSplineInterpolator,
)
from finceptquantlib.scheduling.daycount import DayCountConvention


class Curve(ABC):
    """
    Abstract base class for yield curves.

    A curve represents interest rates as a function of time or maturity.
    Different curve types (discount, zero, forward) inherit from this base.
    """

    def __init__(
        self,
        reference_date: date,
        tenors: Union[List[float], list[float]],
        values: Union[List[float], list[float]],
        interpolation: str = "log_linear",
        day_count: Union[str, DayCountConvention] = "ACT/365F",
    ):
        """
        Initialize a curve.

        Args:
            reference_date: Curve valuation date
            tenors: Time points in years
            values: Curve values at each tenor
            interpolation: Interpolation method
            day_count: Day count convention
        """
        self.reference_date = reference_date
        self._tenors = [float(t) for t in tenors]
        self._values = [float(v) for v in values]

        # Day count convention
        if isinstance(day_count, str):
            from finceptquantlib.scheduling.daycount import _get_convention
            day_count = _get_convention(day_count)
        self._day_count = day_count

        # Set up interpolator
        self._interpolator = self._create_interpolator(interpolation)
        self._interpolation_method = interpolation

    def _create_interpolator(self, method: str) -> Interpolator:
        """Create the appropriate interpolator."""
        methods = {
            "linear": LinearInterpolator,
            "log_linear": LogLinearInterpolator,
            "loglinear": LogLinearInterpolator,
            "cubic": CubicSplineInterpolator,
            "cubic_spline": CubicSplineInterpolator,
            "spline": CubicSplineInterpolator,
        }

        method_key = method.lower().replace("-", "_").replace(" ", "_")
        if method_key not in methods:
            raise ValueError(f"Unknown interpolation method: {method}")

        interpolator_class = methods[method_key]
        return interpolator_class(self._tenors, self._values)

    @property
    def tenors(self) -> list[float]:
        """Tenor points in years."""
        return self._tenors

    @property
    def values(self) -> list[float]:
        """Values at each tenor."""
        return self._values

    @property
    def day_count(self) -> DayCountConvention:
        """Day count convention."""
        return self._day_count

    def time_to_maturity(self, d: date) -> float:
        """Calculate time from reference date to a date in years."""
        return self._day_count.year_fraction(self.reference_date, d)

    def date_from_tenor(self, tenor: float) -> date:
        """Convert tenor in years to a date."""
        days = int(tenor * 365)  # Approximate
        return self.reference_date + timedelta(days=days)

    @abstractmethod
    def discount_factor(
        self,
        t: Union[float, date, list[float]]
    ) -> Union[float, list[float]]:
        """
        Get discount factor for a given maturity.

        Args:
            t: Time in years or date

        Returns:
            Discount factor(s)
        """

    @abstractmethod
    def zero_rate(
        self,
        t: Union[float, date, list[float]],
        compounding: str = "continuous"
    ) -> Union[float, list[float]]:
        """
        Get zero (spot) rate for a given maturity.

        Args:
            t: Time in years or date
            compounding: "continuous", "annual", "semi-annual", "quarterly"

        Returns:
            Zero rate(s)
        """

    def forward_rate(
        self,
        t1: Union[float, date],
        t2: Union[float, date],
        compounding: str = "continuous"
    ) -> float:
        """
        Calculate forward rate between two tenors.

        Args:
            t1: Start time in years or date
            t2: End time in years or date
            compounding: Compounding frequency

        Returns:
            Forward rate

        Examples:
            >>> from datetime import date
            >>> curve = ZeroCurve(date(2024, 1, 15), [1, 2, 5], [0.040, 0.041, 0.042])
            >>> round(curve.forward_rate(1.0, 2.0), 6)  # 1y1y forward
            0.042
        """
        if isinstance(t1, date):
            t1 = self.time_to_maturity(t1)
        if isinstance(t2, date):
            t2 = self.time_to_maturity(t2)

        if t2 <= t1:
            raise ValueError("t2 must be greater than t1")

        df1 = self.discount_factor(t1)
        df2 = self.discount_factor(t2)
        dt = t2 - t1

        if compounding == "continuous":
            return -math.log(df2 / df1) / dt
        else:
            periods_per_year = {
                "annual": 1,
                "semi-annual": 2,
                "quarterly": 4,
                "monthly": 12,
            }.get(compounding, 1)
            return periods_per_year * ((df1 / df2) ** (1 / (periods_per_year * dt)) - 1)

    def instantaneous_forward(self, t: Union[float, date]) -> float:
        """
        Calculate instantaneous forward rate at a given maturity.

        f(t) = -d/dt[ln(DF(t))]

        Args:
            t: Time in years or date

        Returns:
            Instantaneous forward rate
        """
        if isinstance(t, date):
            t = self.time_to_maturity(t)

        # Symmetric central difference
        epsilon = 1e-6
        if t > epsilon:
            df_plus = self.discount_factor(t + epsilon)
            df_minus = self.discount_factor(t - epsilon)
            return -math.log(df_plus / df_minus) / (2 * epsilon)
        else:
            # Near zero: forward difference
            df0 = self.discount_factor(0.0)
            df_eps = self.discount_factor(2 * epsilon)
            return -math.log(df_eps / df0) / (2 * epsilon)

    def shift(self, basis_points: float) -> Curve:
        """
        Create a parallel-shifted curve.

        Args:
            basis_points: Shift amount in basis points

        Returns:
            New shifted curve
        """
        shift = basis_points / 10000.0
        shifted_values = [v + shift for v in self._values]
        return self.__class__(
            reference_date=self.reference_date,
            tenors=self._tenors,
            values=shifted_values,
            interpolation=self._interpolation_method,
            day_count=self._day_count,
        )

    def twist(self, short_bp: float, long_bp: float, pivot_tenor: float = 5.0) -> Curve:
        """
        Create a twisted curve (short and long end move differently).

        Args:
            short_bp: Short end shift in basis points
            long_bp: Long end shift in basis points
            pivot_tenor: Pivot point in years

        Returns:
            New twisted curve
        """
        # Calculate weights clipped to [-1, 1]
        weights = [max(-1.0, min(1.0, (t - pivot_tenor) / pivot_tenor)) for t in self._tenors]
        # Linear interpolation: short_bp at short end, long_bp at long end
        shifts = [(short_bp * (1 - (w + 1) / 2) + long_bp * ((w + 1) / 2)) / 10000.0 for w in weights]
        twisted_values = [v + s for v, s in zip(self._values, shifts)]

        return self.__class__(
            reference_date=self.reference_date,
            tenors=self._tenors,
            values=twisted_values,
            interpolation=self._interpolation_method,
            day_count=self._day_count,
        )

    def roll(self, days: int) -> Curve:
        """
        Roll the curve forward in time.

        Args:
            days: Number of days to roll

        Returns:
            New rolled curve
        """
        new_reference = self.reference_date + timedelta(days=days)

        # Adjust tenors
        day_fraction = days / 365.0
        new_tenors = [t - day_fraction for t in self._tenors]

        # Filter tenors > 0 and corresponding values
        filtered = [(t, v) for t, v in zip(new_tenors, self._values) if t > 0]
        if not filtered:
            raise ValueError(
                f"Rolling curve {days} days leaves no positive tenors. "
                f"Maximum roll is {int(min(self._tenors) * 365)} days."
            )

        new_tenors_filtered, new_values_filtered = zip(*filtered)

        return self.__class__(
            reference_date=new_reference,
            tenors=list(new_tenors_filtered),
            values=list(new_values_filtered),
            interpolation=self._interpolation_method,
            day_count=self._day_count,
        )

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"ref_date={self.reference_date}, "
                f"tenors={len(self._tenors)}, "
                f"interp='{self._interpolation_method}')")


class DiscountCurve(Curve):
    """
    Discount factor curve.

    Stores discount factors directly. Good for discounting cash flows.
    DF(t) = exp(-r(t) * t) where r(t) is the zero rate.
    """

    def discount_factor(
        self,
        t: Union[float, date, list[float]]
    ) -> Union[float, list[float]]:
        """Get discount factor at time t."""
        if isinstance(t, date):
            t = self.time_to_maturity(t)

        if isinstance(t, (int, float)):
            if t <= 0:
                return 1.0
            return float(self._interpolator(t))
        else:
            result = []
            for ti in t:
                if ti <= 0:
                    result.append(1.0)
                else:
                    result.append(float(self._interpolator(ti)))
            return result

    def zero_rate(
        self,
        t: Union[float, date, list[float]],
        compounding: str = "continuous"
    ) -> Union[float, list[float]]:
        """Convert discount factor to zero rate."""
        if isinstance(t, date):
            t = self.time_to_maturity(t)

        df = self.discount_factor(t)

        if compounding == "continuous":
            if isinstance(t, (int, float)):
                return -math.log(df) / t if t > 0 else 0.0
            else:
                result = []
                for ti, dfi in zip(t, df):
                    if ti > 0:
                        result.append(-math.log(dfi) / ti)
                    else:
                        result.append(0.0)
                return result
        else:
            periods_per_year = {
                "annual": 1,
                "semi-annual": 2,
                "quarterly": 4,
                "monthly": 12,
            }.get(compounding, 1)

            if isinstance(t, (int, float)):
                if t <= 0:
                    return 0.0
                return periods_per_year * (df ** (-1 / (periods_per_year * t)) - 1)
            else:
                result = []
                for ti, dfi in zip(t, df):
                    if ti <= 0:
                        result.append(0.0)
                    else:
                        result.append(periods_per_year * (dfi ** (-1 / (periods_per_year * ti)) - 1))
                return result


class ZeroCurve(Curve):
    """
    Zero (spot) rate curve.

    Stores continuously compounded zero rates.
    """

    def __init__(
        self,
        reference_date: date,
        tenors: Union[List[float], list[float]],
        values: Union[List[float], list[float]],
        interpolation: str = "linear",
        day_count: Union[str, DayCountConvention] = "ACT/365F",
    ):
        """
        Initialize zero rate curve.

        Args:
            reference_date: Curve valuation date
            tenors: Time points in years
            values: Zero rates at each tenor (as decimals, e.g., 0.05 for 5%)
            interpolation: Interpolation method
            day_count: Day count convention
        """
        super().__init__(
            reference_date=reference_date,
            tenors=tenors,
            values=values,
            interpolation=interpolation,
            day_count=day_count,
        )

    def zero_rate(
        self,
        t: Union[float, date, list[float]],
        compounding: str = "continuous"
    ) -> Union[float, list[float]]:
        """Get zero rate at time t (continuous compounding assumed in storage)."""
        if isinstance(t, date):
            t = self.time_to_maturity(t)

        if isinstance(t, (int, float)):
            if t <= 0:
                return float(self._values[0]) if len(self._values) > 0 else 0.0
            rate = float(self._interpolator(t))
        else:
            rate = []
            for ti in t:
                if ti <= 0:
                    rate.append(float(self._values[0]) if len(self._values) > 0 else 0.0)
                else:
                    rate.append(float(self._interpolator(ti)))

        if compounding == "continuous":
            return rate
        else:
            # Convert from continuous to discrete compounding
            periods_per_year = {
                "annual": 1,
                "semi-annual": 2,
                "quarterly": 4,
                "monthly": 12,
            }.get(compounding, 1)
            if isinstance(rate, (int, float)):
                return periods_per_year * (math.exp(rate / periods_per_year) - 1)
            else:
                return [periods_per_year * (math.exp(r / periods_per_year) - 1) for r in rate]

    def discount_factor(
        self,
        t: Union[float, date, list[float]]
    ) -> Union[float, list[float]]:
        """Convert zero rate to discount factor."""
        if isinstance(t, date):
            t = self.time_to_maturity(t)

        rate = self.zero_rate(t, compounding="continuous")

        if isinstance(t, (int, float)):
            return math.exp(-rate * t) if t > 0 else 1.0
        else:
            result = []
            for ti, ri in zip(t, rate):
                if ti > 0:
                    result.append(math.exp(-ri * ti))
                else:
                    result.append(1.0)
            return result


class ForwardCurve(Curve):
    """
    Instantaneous forward rate curve.

    Stores instantaneous forward rates.
    f(t) = -d/dt[ln(DF(t))]
    """

    def __init__(
        self,
        reference_date: date,
        tenors: Union[List[float], list[float]],
        values: Union[List[float], list[float]],
        interpolation: str = "linear",
        day_count: Union[str, DayCountConvention] = "ACT/365F",
    ):
        """
        Initialize forward rate curve.

        Args:
            reference_date: Curve valuation date
            tenors: Time points in years
            values: Forward rates at each tenor
            interpolation: Interpolation method
            day_count: Day count convention
        """
        super().__init__(
            reference_date=reference_date,
            tenors=tenors,
            values=values,
            interpolation=interpolation,
            day_count=day_count,
        )

    def forward_rate_at(self, t: Union[float, date]) -> float:
        """Get instantaneous forward rate at time t."""
        if isinstance(t, date):
            t = self.time_to_maturity(t)

        if t <= 0:
            return float(self._values[0]) if len(self._values) > 0 else 0.0

        return float(self._interpolator(t))

    def zero_rate(
        self,
        t: Union[float, date, list[float]],
        compounding: str = "continuous"
    ) -> Union[float, list[float]]:
        """
        Convert forward rates to zero rates via integration.

        r(T) = (1/T) * integral_0^T f(s) ds
        """
        if isinstance(t, date):
            t = self.time_to_maturity(t)

        if isinstance(t, (int, float)):
            if t <= 0:
                return 0.0
            # Numerical integration using trapezoidal rule
            n_points = max(10, int(t * 100))
            step = t / (n_points - 1)
            s = [i * step for i in range(n_points)]
            f_values = [self.forward_rate_at(si) for si in s]
            # Trapezoidal integration
            integral = sum((f_values[i] + f_values[i+1]) / 2 * (s[i+1] - s[i]) for i in range(len(s) - 1))
            rate = integral / t
        else:
            rate = [self.zero_rate(ti) for ti in t]

        if compounding == "continuous":
            return rate
        else:
            periods_per_year = {
                "annual": 1,
                "semi-annual": 2,
                "quarterly": 4,
                "monthly": 12,
            }.get(compounding, 1)
            if isinstance(rate, (int, float)):
                return periods_per_year * (math.exp(rate / periods_per_year) - 1)
            else:
                return [periods_per_year * (math.exp(r / periods_per_year) - 1) for r in rate]

    def discount_factor(
        self,
        t: Union[float, date, list[float]]
    ) -> Union[float, list[float]]:
        """
        Convert forward rates to discount factors.

        DF(T) = exp(-integral_0^T f(s) ds)
        """
        if isinstance(t, date):
            t = self.time_to_maturity(t)

        if isinstance(t, (int, float)):
            if t <= 0:
                return 1.0
            rate = self.zero_rate(t)
            return math.exp(-rate * t)
        else:
            rates = self.zero_rate(t)
            result = []
            for ti, ri in zip(t, rates):
                if ti > 0:
                    result.append(math.exp(-ri * ti))
                else:
                    result.append(1.0)
            return result
