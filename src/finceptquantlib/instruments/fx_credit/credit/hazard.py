"""
Hazard Rate and Survival Probability Models

Implements:
- Survival curve representations
- Hazard rate models (constant, piecewise)
- Default probability calculations

The hazard rate (λ) is the instantaneous probability of default,
conditional on survival to that point:

λ(t) = -d/dt[ln(S(t))]

where S(t) is the survival probability to time t.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Union, List
import math
import bisect

from finceptquantlib.curves.interpolation import LinearInterpolator, LogLinearInterpolator
from finceptquantlib.scheduling.daycount import DayCountConvention, _get_convention


class SurvivalCurve(ABC):
    """
    Abstract base class for survival probability curves.

    Survival probability S(t) is the probability of no default by time t.
    """

    @abstractmethod
    def survival_probability(
        self,
        t: Union[float, date, list[float]]
    ) -> Union[float, list[float]]:
        """
        Get survival probability to time t.

        Args:
            t: Time in years or date

        Returns:
            Survival probability (between 0 and 1)
        """

    @abstractmethod
    def default_probability(
        self,
        t: Union[float, date, list[float]]
    ) -> Union[float, list[float]]:
        """
        Get cumulative default probability to time t.

        P(τ ≤ t) = 1 - S(t)
        """

    def marginal_default_probability(
        self,
        t1: Union[float, date],
        t2: Union[float, date],
    ) -> float:
        """
        Get marginal default probability between t1 and t2.

        P(t1 < τ ≤ t2 | τ > t1) = (S(t1) - S(t2)) / S(t1)
        """
        s1 = self.survival_probability(t1)
        s2 = self.survival_probability(t2)

        if s1 <= 0:
            return 1.0

        return (s1 - s2) / s1

    @abstractmethod
    def hazard_rate(
        self,
        t: Union[float, date]
    ) -> float:
        """
        Get instantaneous hazard rate at time t.

        λ(t) = -d/dt[ln(S(t))]
        """


class HazardRateCurve(SurvivalCurve):
    """
    Survival curve defined by a term structure of hazard rates.

    Given hazard rates λ(t), survival probability is:
    S(T) = exp(-∫₀ᵀ λ(t) dt)
    """

    def __init__(
        self,
        reference_date: date,
        tenors: Union[List[float], list[float]],
        hazard_rates: Union[List[float], list[float]],
        interpolation: str = "linear",
        day_count: Union[str, DayCountConvention] = "ACT/365F",
    ):
        """
        Initialize hazard rate curve.

        Args:
            reference_date: Curve valuation date
            tenors: Time points in years
            hazard_rates: Hazard rates at each tenor
            interpolation: Interpolation method
            day_count: Day count convention
        """
        self.reference_date = reference_date
        self._tenors = [float(t) for t in tenors]
        self._hazard_rates = [float(h) for h in hazard_rates]

        if isinstance(day_count, str):
            day_count = _get_convention(day_count)
        self._day_count = day_count

        # Create interpolator
        if interpolation.lower() in ["linear", "flat"]:
            self._interpolator = LinearInterpolator(self._tenors, self._hazard_rates)
        else:
            self._interpolator = LogLinearInterpolator(self._tenors, self._hazard_rates)

    def _to_year_fraction(self, t: Union[float, date]) -> float:
        """Convert date to year fraction."""
        if isinstance(t, date):
            return self._day_count.year_fraction(self.reference_date, t)
        return float(t)

    def hazard_rate(self, t: Union[float, date]) -> float:
        """Get hazard rate at time t."""
        t_yf = self._to_year_fraction(t)

        if t_yf <= 0:
            return float(self._hazard_rates[0]) if len(self._hazard_rates) > 0 else 0.0

        return float(self._interpolator(t_yf))

    def survival_probability(
        self,
        t: Union[float, date, list[float]]
    ) -> Union[float, list[float]]:
        """
        Calculate survival probability using numerical integration.

        S(T) = exp(-integral_0^T lambda(t) dt)
        """
        if isinstance(t, date):
            t = self._to_year_fraction(t)

        scalar_input = isinstance(t, (int, float))
        if scalar_input:
            t_list = [float(t)]
        else:
            t_list = [float(ti) for ti in t]

        results = []

        for ti in t_list:
            if ti <= 0:
                results.append(1.0)
            else:
                # Numerical integration using trapezoidal rule
                n_points = max(10, int(ti * 50))
                s = [ti * i / (n_points - 1) for i in range(n_points)]
                lambda_values = [self.hazard_rate(si) for si in s]
                # Trapezoidal integration
                integral = 0.0
                for i in range(len(s) - 1):
                    integral += 0.5 * (lambda_values[i] + lambda_values[i + 1]) * (s[i + 1] - s[i])
                results.append(math.exp(-integral))

        return results[0] if scalar_input else results

    def default_probability(
        self,
        t: Union[float, date, list[float]]
    ) -> Union[float, list[float]]:
        """Get cumulative default probability."""
        s = self.survival_probability(t)
        if isinstance(s, list):
            return [1.0 - si for si in s]
        return 1.0 - s

    def shift(self, basis_points: float) -> "HazardRateCurve":
        """Create a parallel-shifted hazard curve."""
        shift = basis_points / 10000.0
        shifted_rates = [h + shift for h in self._hazard_rates]
        return HazardRateCurve(
            reference_date=self.reference_date,
            tenors=self._tenors,
            hazard_rates=shifted_rates,
            day_count=self._day_count,
        )

    def __repr__(self) -> str:
        return (f"HazardRateCurve(ref_date={self.reference_date}, "
                f"tenors={len(self._tenors)})")


class ConstantHazardRate(SurvivalCurve):
    """
    Constant hazard rate model.

    The simplest hazard model where λ(t) = λ for all t.

    Implies exponential survival: S(t) = exp(-λt)
    """

    def __init__(
        self,
        reference_date: date,
        hazard_rate: float,
        day_count: Union[str, DayCountConvention] = "ACT/365F",
    ):
        """
        Initialize constant hazard rate model.

        Args:
            reference_date: Reference date
            hazard_rate: Constant hazard rate (annualized)
            day_count: Day count convention
        """
        self.reference_date = reference_date
        self._hazard_rate = hazard_rate

        if isinstance(day_count, str):
            day_count = _get_convention(day_count)
        self._day_count = day_count

    def _to_year_fraction(self, t: Union[float, date]) -> float:
        if isinstance(t, date):
            return self._day_count.year_fraction(self.reference_date, t)
        return float(t)

    def hazard_rate(self, t: Union[float, date]) -> float:
        """Constant hazard rate."""
        return self._hazard_rate

    def survival_probability(
        self,
        t: Union[float, date, list[float]]
    ) -> Union[float, list[float]]:
        """S(t) = exp(-lambda*t)"""
        if isinstance(t, date):
            t = self._to_year_fraction(t)

        scalar_input = isinstance(t, (int, float))
        if scalar_input:
            t_list = [max(float(t), 0)]
        else:
            t_list = [max(float(ti), 0) for ti in t]

        result = [math.exp(-self._hazard_rate * ti) for ti in t_list]

        return result[0] if scalar_input else result

    def default_probability(
        self,
        t: Union[float, date, list[float]]
    ) -> Union[float, list[float]]:
        """P(tau <= t) = 1 - exp(-lambda*t)"""
        s = self.survival_probability(t)
        if isinstance(s, list):
            return [1.0 - si for si in s]
        return 1.0 - s

    @classmethod
    def from_spread(
        cls,
        reference_date: date,
        cds_spread: float,
        recovery_rate: float = 0.4,
    ) -> "ConstantHazardRate":
        """
        Create from CDS spread.

        Approximate relationship:
        Spread ≈ λ * (1 - R)
        => λ ≈ Spread / (1 - R)

        Args:
            reference_date: Reference date
            cds_spread: CDS spread (annualized, in decimal)
            recovery_rate: Recovery rate assumption

        Returns:
            ConstantHazardRate
        """
        if recovery_rate >= 1:
            raise ValueError("Recovery rate must be less than 1")

        hazard_rate = cds_spread / (1 - recovery_rate)
        return cls(reference_date, hazard_rate)

    def __repr__(self) -> str:
        return f"ConstantHazardRate(λ={self._hazard_rate:.4%})"


class PiecewiseHazardRate(SurvivalCurve):
    """
    Piecewise constant hazard rate model.

    Hazard rate is constant between breakpoints:
    λ(t) = λᵢ for tᵢ₋₁ < t ≤ tᵢ

    This is the standard model for CDS bootstrapping.
    """

    def __init__(
        self,
        reference_date: date,
        tenors: Union[List[float], list[float]],
        hazard_rates: Union[List[float], list[float]],
        day_count: Union[str, DayCountConvention] = "ACT/365F",
    ):
        """
        Initialize piecewise hazard model.

        Args:
            reference_date: Reference date
            tenors: Breakpoint tenors (right endpoints)
            hazard_rates: Hazard rate for each period
            day_count: Day count convention
        """
        self.reference_date = reference_date
        self._tenors = [float(t) for t in tenors]
        self._hazard_rates = [float(h) for h in hazard_rates]

        if isinstance(day_count, str):
            day_count = _get_convention(day_count)
        self._day_count = day_count

        # Precompute cumulative survival for efficiency
        self._precompute_survival()

    def _precompute_survival(self):
        """Precompute survival probabilities at breakpoints."""
        n = len(self._tenors)
        self._survival_at_tenors = [1.0] * (n + 1)

        prev_t = 0.0
        for i in range(n):
            dt = self._tenors[i] - prev_t
            self._survival_at_tenors[i + 1] = (
                self._survival_at_tenors[i] * math.exp(-self._hazard_rates[i] * dt)
            )
            prev_t = self._tenors[i]

    def _to_year_fraction(self, t: Union[float, date]) -> float:
        if isinstance(t, date):
            return self._day_count.year_fraction(self.reference_date, t)
        return float(t)

    def hazard_rate(self, t: Union[float, date]) -> float:
        """Get hazard rate at time t (piecewise constant)."""
        t_yf = self._to_year_fraction(t)

        if t_yf <= 0:
            return float(self._hazard_rates[0])

        # Find which segment t falls into
        idx = bisect.bisect_left(self._tenors, t_yf)
        if idx >= len(self._hazard_rates):
            return float(self._hazard_rates[-1])

        return float(self._hazard_rates[idx])

    def survival_probability(
        self,
        t: Union[float, date, list[float]]
    ) -> Union[float, list[float]]:
        """Calculate survival probability."""
        if isinstance(t, date):
            t = self._to_year_fraction(t)

        scalar_input = isinstance(t, (int, float))
        if scalar_input:
            t_list = [float(t)]
        else:
            t_list = [float(ti) for ti in t]

        results = []

        for ti in t_list:
            if ti <= 0:
                results.append(1.0)
            else:
                # Find segment
                idx = bisect.bisect_left(self._tenors, ti)
                if idx == 0:
                    results.append(math.exp(-self._hazard_rates[0] * ti))
                elif idx >= len(self._tenors):
                    # Extrapolate with last hazard rate
                    prev_t = self._tenors[-1]
                    dt = ti - prev_t
                    results.append(self._survival_at_tenors[-1] * math.exp(-self._hazard_rates[-1] * dt))
                else:
                    prev_t = self._tenors[idx - 1]
                    dt = ti - prev_t
                    results.append(self._survival_at_tenors[idx] * math.exp(-self._hazard_rates[idx] * dt))

        return results[0] if scalar_input else results

    def default_probability(
        self,
        t: Union[float, date, list[float]]
    ) -> Union[float, list[float]]:
        """Get cumulative default probability."""
        s = self.survival_probability(t)
        if isinstance(s, list):
            return [1.0 - si for si in s]
        return 1.0 - s

    def shift(self, basis_points: float) -> "PiecewiseHazardRate":
        """Create parallel-shifted curve."""
        shift = basis_points / 10000.0
        shifted = PiecewiseHazardRate(
            reference_date=self.reference_date,
            tenors=self._tenors.copy(),
            hazard_rates=[h + shift for h in self._hazard_rates],
            day_count=self._day_count,
        )
        return shifted

    def __repr__(self) -> str:
        return (f"PiecewiseHazardRate(tenors={len(self._tenors)}, "
                f"lambda_range=[{min(self._hazard_rates):.4%}, {max(self._hazard_rates):.4%}])")
