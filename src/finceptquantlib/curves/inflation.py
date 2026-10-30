"""
Inflation Curves

Provides:
- CPI/inflation index curves
- Real rate curves
- Seasonality adjustments
- Inflation swap pricing support
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, List
import math
import bisect

from finceptquantlib.curves.curve import Curve
from finceptquantlib.curves.interpolation import LinearInterpolator, CubicSplineInterpolator


@dataclass
class CPIObservation:
    """A CPI index observation."""
    observation_date: date
    value: float


class InflationCurve(Curve):
    """
    Inflation index curve.

    Models the expected future CPI index levels.
    """

    def __init__(
        self,
        reference_date: date,
        base_cpi: float,
        tenors: list[float],
        inflation_rates: list[float],  # Annual inflation rates
        interpolation: str = "linear",
        seasonality: Optional[list[float]] = None,  # 12 monthly factors
    ):
        self.reference_date = reference_date
        self._base_cpi = base_cpi
        self._tenors = [float(t) for t in tenors]
        self._inflation_rates = [float(r) for r in inflation_rates]
        self._interpolation_method = interpolation
        self._seasonality = seasonality

        # Build CPI index levels
        self._cpi_levels = self._build_cpi_levels()

        if interpolation == "linear":
            self._interp = LinearInterpolator(self._tenors, self._cpi_levels)
        else:
            self._interp = CubicSplineInterpolator(self._tenors, self._cpi_levels)

    def _build_cpi_levels(self) -> list[float]:
        """Build CPI index levels from inflation rates."""
        levels = [0.0] * len(self._tenors)

        for i, t in enumerate(self._tenors):
            # CPI(t) = CPI(0) * (1 + π)^t
            # But with term structure: use forward inflation
            if i == 0:
                levels[i] = self._base_cpi * (1 + self._inflation_rates[i]) ** t
            else:
                # Chain forward inflation rates
                prev_level = levels[i - 1]
                prev_t = self._tenors[i - 1]
                fwd_rate = self._inflation_rates[i]
                levels[i] = prev_level * (1 + fwd_rate) ** (t - prev_t)

        return levels

    def cpi_index(self, t: float) -> float:
        """
        Get expected CPI index at time t.

        Args:
            t: Time in years from reference date

        Returns:
            Expected CPI index level
        """
        if t <= 0:
            return self._base_cpi

        if t <= self._tenors[0]:
            return self._base_cpi * (1 + self._inflation_rates[0]) ** t

        if t >= self._tenors[-1]:
            # Extrapolate with last rate
            last_cpi = self._cpi_levels[-1]
            last_t = self._tenors[-1]
            return last_cpi * (1 + self._inflation_rates[-1]) ** (t - last_t)

        return float(self._interp(t))

    def cpi_index_on_date(self, target_date: date) -> float:
        """Get expected CPI index for a specific date."""
        t = (target_date - self.reference_date).days / 365.0
        cpi = self.cpi_index(t)

        # Apply seasonality if available
        if self._seasonality is not None:
            month = target_date.month - 1  # 0-indexed
            cpi *= self._seasonality[month]

        return cpi

    def inflation_rate(self, t: float) -> float:
        """
        Get instantaneous inflation rate at time t.

        Args:
            t: Time in years

        Returns:
            Annual inflation rate
        """
        if t <= self._tenors[0]:
            return float(self._inflation_rates[0])
        if t >= self._tenors[-1]:
            return float(self._inflation_rates[-1])

        # Interpolate
        idx = bisect.bisect_right(self._tenors, t)
        if idx == 0:
            return float(self._inflation_rates[0])

        t1, t2 = self._tenors[idx - 1], self._tenors[idx]
        r1, r2 = self._inflation_rates[idx - 1], self._inflation_rates[idx]

        w = (t - t1) / (t2 - t1)
        return float(r1 * (1 - w) + r2 * w)

    def forward_inflation(self, t1: float, t2: float) -> float:
        """
        Get forward inflation rate between t1 and t2.

        Args:
            t1: Start time
            t2: End time

        Returns:
            Annual forward inflation rate
        """
        if t2 <= t1:
            return 0.0

        cpi1 = self.cpi_index(t1)
        cpi2 = self.cpi_index(t2)

        return (cpi2 / cpi1) ** (1 / (t2 - t1)) - 1

    def zero_rate(self, t, compounding: str = "continuous"):
        """Zero inflation rate (implements Curve abstract method)."""
        if isinstance(t, date):
            t = (t - self.reference_date).days / 365.0
        if isinstance(t, (int, float)):
            return self.zero_inflation_rate(t)
        return [self.zero_inflation_rate(ti) for ti in t]

    def discount_factor(self, t):
        """Real discount factor (implements Curve abstract method)."""
        if isinstance(t, date):
            t = (t - self.reference_date).days / 365.0
        if isinstance(t, (int, float)):
            r = self.zero_inflation_rate(t) if t > 0 else 0.0
            return math.exp(-r * t) if t > 0 else 1.0
        return [self.discount_factor(ti) for ti in t]

    def zero_inflation_rate(self, t: float) -> float:
        """
        Get zero-coupon inflation rate to time t.

        Args:
            t: Time in years

        Returns:
            Zero inflation rate
        """
        if t <= 0:
            return float(self._inflation_rates[0])

        cpi_t = self.cpi_index(t)
        return (cpi_t / self._base_cpi) ** (1 / t) - 1

    def real_discount_factor(
        self,
        t: float,
        nominal_curve: Curve,
    ) -> float:
        """
        Get real (inflation-adjusted) discount factor.

        DF_real = DF_nominal * CPI(0) / CPI(t)
        """
        df_nominal = nominal_curve.discount_factor(t)
        cpi_ratio = self._base_cpi / self.cpi_index(t)
        return df_nominal * cpi_ratio


class RealRateCurve(Curve):
    """
    Real (inflation-adjusted) interest rate curve.

    Built from nominal curve and inflation expectations.
    """

    def __init__(
        self,
        reference_date: date,
        nominal_curve: Curve,
        inflation_curve: InflationCurve,
    ):
        self.reference_date = reference_date
        self._nominal_curve = nominal_curve
        self._inflation_curve = inflation_curve

    def real_rate(self, t: float) -> float:
        """
        Get real zero rate at time t.

        Uses Fisher equation: (1 + r_nominal) = (1 + r_real)(1 + π)
        """
        nominal_rate = self._nominal_curve.zero_rate(t)
        inflation_rate = self._inflation_curve.zero_inflation_rate(t)

        return (1 + nominal_rate) / (1 + inflation_rate) - 1

    def zero_rate(self, t, compounding: str = "continuous"):
        """Get real zero rate (implements Curve abstract method)."""
        if isinstance(t, date):
            t = (t - self.reference_date).days / 365.0
        if isinstance(t, (int, float)):
            return self.real_rate(t)
        return [self.real_rate(ti) for ti in t]

    def discount_factor(self, t, **kwargs):
        """Get real discount factor."""
        if isinstance(t, date):
            t = (t - self.reference_date).days / 365.0
        if isinstance(t, (int, float)):
            r = self.real_rate(t)
            return math.exp(-r * t) if t > 0 else 1.0
        return [self.discount_factor(ti) for ti in t]


class SeasonalityModel:
    """
    CPI seasonality model.

    Models monthly seasonal patterns in inflation.
    """

    def __init__(
        self,
        monthly_factors: Optional[list[float]] = None,
    ):
        if monthly_factors is not None:
            if len(monthly_factors) != 12:
                raise ValueError("Must provide 12 monthly factors")
            # Normalize to average 1
            mean_factor = sum(monthly_factors) / len(monthly_factors)
            self._factors = [f / mean_factor for f in monthly_factors]
        else:
            # Default: no seasonality
            self._factors = [1.0] * 12

    def factor(self, month: int) -> float:
        """Get seasonality factor for month (1-12)."""
        return float(self._factors[month - 1])

    def adjust(self, cpi: float, month: int) -> float:
        """Apply seasonality adjustment."""
        return cpi * self._factors[month - 1]

    def deseasonalize(self, cpi: float, month: int) -> float:
        """Remove seasonality."""
        return cpi / self._factors[month - 1]

    @classmethod
    def from_historical(cls, observations: List[CPIObservation]) -> "SeasonalityModel":
        """
        Estimate seasonality from historical CPI observations.

        Uses ratio-to-moving-average method.
        """
        if len(observations) < 24:
            return cls()  # Not enough data

        # Sort by date
        sorted_obs = sorted(observations, key=lambda x: x.observation_date)

        # Calculate 12-month moving average
        values = [o.value for o in sorted_obs]
        n = len(values)

        monthly_ratios = {m: [] for m in range(12)}

        for i in range(6, n - 6):
            ma = sum(values[i - 6:i + 6]) / len(values[i - 6:i + 6])
            month = sorted_obs[i].observation_date.month - 1
            monthly_ratios[month].append(values[i] / ma)

        # Average ratios by month
        factors = [
            sum(monthly_ratios[m]) / len(monthly_ratios[m]) if monthly_ratios[m] else 1.0
            for m in range(12)
        ]

        return cls(factors)


def bootstrap_inflation_curve(
    reference_date: date,
    base_cpi: float,
    swap_tenors: List[float],
    swap_rates: List[float],
    nominal_curve: Curve,
) -> InflationCurve:
    """
    Bootstrap inflation curve from zero-coupon inflation swaps.

    Args:
        reference_date: Curve reference date
        base_cpi: Current CPI index level
        swap_tenors: Swap maturities in years
        swap_rates: Quoted inflation swap rates
        nominal_curve: Nominal discount curve

    Returns:
        Bootstrapped InflationCurve
    """
    inflation_rates = []

    for i, (tenor, rate) in enumerate(zip(swap_tenors, swap_rates)):
        # ZC inflation swap: pay (CPI(T)/CPI(0) - 1) at T
        # NPV = 0 => E[CPI(T)] / CPI(0) = (1 + rate)^T
        # So implied inflation rate is approximately the swap rate
        inflation_rates.append(rate)

    return InflationCurve(
        reference_date=reference_date,
        base_cpi=base_cpi,
        tenors=swap_tenors,
        inflation_rates=inflation_rates,
    )
