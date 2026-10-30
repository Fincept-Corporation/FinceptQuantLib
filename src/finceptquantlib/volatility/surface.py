"""
Volatility Surfaces

Provides:
- Flat volatility
- Term structure volatility
- Full volatility surface (strike x expiry)
- Volatility cube (strike x expiry x tenor)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict
import math
import bisect

from finceptquantlib.curves.interpolation import (
    LinearInterpolator,
    CubicSplineInterpolator,
)


class VolatilitySurface(ABC):
    """Abstract base class for volatility surfaces."""

    @abstractmethod
    def volatility(
        self,
        strike: float,
        expiry: float,
        forward: Optional[float] = None,
    ) -> float:
        """Get volatility for given strike and expiry."""

    @abstractmethod
    def total_variance(
        self,
        strike: float,
        expiry: float,
        forward: Optional[float] = None,
    ) -> float:
        """Get total variance (vol^2 * T)."""


class FlatVolatility(VolatilitySurface):
    """Constant volatility across all strikes and expiries."""

    def __init__(self, volatility: float):
        self._vol = volatility

    def volatility(
        self,
        strike: float,
        expiry: float,
        forward: Optional[float] = None,
    ) -> float:
        return self._vol

    def total_variance(
        self,
        strike: float,
        expiry: float,
        forward: Optional[float] = None,
    ) -> float:
        return self._vol ** 2 * expiry


class TermStructureVolatility(VolatilitySurface):
    """Volatility with term structure (ATM vol vs expiry)."""

    def __init__(
        self,
        expiries: list[float],
        volatilities: list[float],
        interpolation: str = "linear",
    ):
        self._expiries = [float(e) for e in expiries]
        self._vols = [float(v) for v in volatilities]

        if interpolation == "linear":
            self._interp = LinearInterpolator(self._expiries, self._vols)
        else:
            self._interp = CubicSplineInterpolator(self._expiries, self._vols)

    def volatility(
        self,
        strike: float,
        expiry: float,
        forward: Optional[float] = None,
    ) -> float:
        # Ignores strike - ATM vol only
        if expiry <= self._expiries[0]:
            return float(self._vols[0])
        if expiry >= self._expiries[-1]:
            return float(self._vols[-1])
        return float(self._interp(expiry))

    def total_variance(
        self,
        strike: float,
        expiry: float,
        forward: Optional[float] = None,
    ) -> float:
        vol = self.volatility(strike, expiry, forward)
        return vol ** 2 * expiry


class VolatilitySurfaceGrid(VolatilitySurface):
    """
    Full volatility surface on a strike-expiry grid.

    Uses bilinear or bicubic interpolation.
    """

    def __init__(
        self,
        strikes: list[float],
        expiries: list[float],
        volatilities: list[list[float]],  # Shape: (n_expiries, n_strikes)
        strike_type: str = "absolute",  # "absolute", "moneyness", "delta"
        interpolation: str = "bilinear",
    ):
        self._strikes = [float(k) for k in strikes]
        self._expiries = [float(e) for e in expiries]
        self._vols = [[float(v) for v in row] for row in volatilities]
        self._strike_type = strike_type
        self._interpolation = interpolation

        if len(self._vols) != len(self._expiries) or (self._vols and len(self._vols[0]) != len(self._strikes)):
            raise ValueError(
                f"Volatility matrix shape ({len(self._vols)}, {len(self._vols[0]) if self._vols else 0}) doesn't match "
                f"({len(self._expiries)}, {len(self._strikes)})"
            )

    def volatility(
        self,
        strike: float,
        expiry: float,
        forward: Optional[float] = None,
    ) -> float:
        # Convert strike if needed
        if self._strike_type == "moneyness" and forward is not None:
            x = strike / forward
        elif self._strike_type == "log_moneyness" and forward is not None:
            x = math.log(strike / forward)
        else:
            x = strike

        return self._bilinear_interp(x, expiry)

    def _bilinear_interp(self, strike: float, expiry: float) -> float:
        """Bilinear interpolation on the grid."""
        # Find bounding indices
        i_exp = bisect.bisect_right(self._expiries, expiry) - 1
        j_strike = bisect.bisect_right(self._strikes, strike) - 1

        # Clamp to valid range
        i_exp = max(0, min(i_exp, len(self._expiries) - 2))
        j_strike = max(0, min(j_strike, len(self._strikes) - 2))

        # Interpolation weights
        t1, t2 = self._expiries[i_exp], self._expiries[i_exp + 1]
        k1, k2 = self._strikes[j_strike], self._strikes[j_strike + 1]

        wt = (expiry - t1) / (t2 - t1) if t2 != t1 else 0
        wk = (strike - k1) / (k2 - k1) if k2 != k1 else 0

        wt = max(0, min(1, wt))
        wk = max(0, min(1, wk))

        # Bilinear interpolation
        v00 = self._vols[i_exp][j_strike]
        v01 = self._vols[i_exp][j_strike + 1]
        v10 = self._vols[i_exp + 1][j_strike]
        v11 = self._vols[i_exp + 1][j_strike + 1]

        vol = (
            (1 - wt) * (1 - wk) * v00 +
            (1 - wt) * wk * v01 +
            wt * (1 - wk) * v10 +
            wt * wk * v11
        )

        return float(vol)

    def total_variance(
        self,
        strike: float,
        expiry: float,
        forward: Optional[float] = None,
    ) -> float:
        vol = self.volatility(strike, expiry, forward)
        return vol ** 2 * expiry

    def smile(self, expiry: float, forward: Optional[float] = None) -> Dict[float, float]:
        """Get volatility smile for a specific expiry."""
        return {
            k: self.volatility(k, expiry, forward)
            for k in self._strikes
        }


class VolatilityCube(VolatilitySurface):
    """
    Volatility cube for swaptions.

    Three dimensions:
    - Option expiry
    - Underlying swap tenor
    - Strike (or moneyness)
    """

    def __init__(
        self,
        expiries: list[float],
        tenors: list[float],
        strikes: list[float],
        volatilities: list[list[list[float]]],  # Shape: (n_expiries, n_tenors, n_strikes)
    ):
        self._expiries = [float(e) for e in expiries]
        self._tenors = [float(t) for t in tenors]
        self._strikes = [float(k) for k in strikes]
        self._vols = [[[float(v) for v in row] for row in plane] for plane in volatilities]

        expected_shape = (len(self._expiries), len(self._tenors), len(self._strikes))
        actual_shape = (len(self._vols), len(self._vols[0]) if self._vols else 0, len(self._vols[0][0]) if self._vols and self._vols[0] else 0)
        if actual_shape != expected_shape:
            raise ValueError(
                f"Volatility cube shape {actual_shape} doesn't match {expected_shape}"
            )

    def volatility(
        self,
        strike: float,
        expiry: float,
        forward: Optional[float] = None,
        tenor: Optional[float] = None,
    ) -> float:
        """
        Get volatility from cube.

        Args:
            strike: Strike rate
            expiry: Option expiry
            forward: Forward rate (optional, for moneyness)
            tenor: Underlying swap tenor (required for cube)
        """
        if tenor is None:
            raise ValueError("Tenor required for volatility cube")

        return self._trilinear_interp(strike, expiry, tenor)

    def _trilinear_interp(
        self,
        strike: float,
        expiry: float,
        tenor: float,
    ) -> float:
        """Trilinear interpolation in the cube."""
        # Find indices
        i_exp = bisect.bisect_right(self._expiries, expiry) - 1
        j_ten = bisect.bisect_right(self._tenors, tenor) - 1
        k_str = bisect.bisect_right(self._strikes, strike) - 1

        # Clamp
        i_exp = max(0, min(i_exp, len(self._expiries) - 2))
        j_ten = max(0, min(j_ten, len(self._tenors) - 2))
        k_str = max(0, min(k_str, len(self._strikes) - 2))

        # Weights
        t1, t2 = self._expiries[i_exp], self._expiries[i_exp + 1]
        n1, n2 = self._tenors[j_ten], self._tenors[j_ten + 1]
        k1, k2 = self._strikes[k_str], self._strikes[k_str + 1]

        wt = (expiry - t1) / (t2 - t1) if t2 != t1 else 0
        wn = (tenor - n1) / (n2 - n1) if n2 != n1 else 0
        wk = (strike - k1) / (k2 - k1) if k2 != k1 else 0

        wt = max(0, min(1, wt))
        wn = max(0, min(1, wn))
        wk = max(0, min(1, wk))

        # Trilinear interpolation
        vol = 0.0
        for di in [0, 1]:
            for dj in [0, 1]:
                for dk in [0, 1]:
                    wi = wt if di else (1 - wt)
                    wj = wn if dj else (1 - wn)
                    wkk = wk if dk else (1 - wk)
                    vol += wi * wj * wkk * self._vols[i_exp + di][j_ten + dj][k_str + dk]

        return float(vol)

    def total_variance(
        self,
        strike: float,
        expiry: float,
        forward: Optional[float] = None,
        tenor: Optional[float] = None,
    ) -> float:
        vol = self.volatility(strike, expiry, forward, tenor)
        return vol ** 2 * expiry

    def atm_surface(self, atm_strike: float) -> list[list[float]]:
        """Get ATM volatility surface (expiry x tenor)."""
        # Find closest strike index
        k_idx = min(range(len(self._strikes)), key=lambda i: abs(self._strikes[i] - atm_strike))
        return [[self._vols[i][j][k_idx] for j in range(len(self._tenors))] for i in range(len(self._expiries))]


@dataclass
class VolatilityPoint:
    """A single volatility observation."""
    expiry: float
    strike: float
    volatility: float
    tenor: Optional[float] = None
    forward: Optional[float] = None
    weight: float = 1.0


def build_surface_from_points(
    points: List[VolatilityPoint],
    expiry_grid: Optional[list[float]] = None,
    strike_grid: Optional[list[float]] = None,
) -> VolatilitySurfaceGrid:
    """
    Build a volatility surface from scattered points.

    Uses least squares fitting to a grid.
    """
    if expiry_grid is None:
        expiry_grid = sorted({p.expiry for p in points})

    if strike_grid is None:
        strike_grid = sorted({p.strike for p in points})

    # Initialize with average vol
    avg_vol = sum(p.volatility for p in points) / len(points) if points else 0.0
    vol_matrix = [[avg_vol for _ in strike_grid] for _ in expiry_grid]

    # Fill in known points (nearest neighbor for now)
    for point in points:
        i = min(range(len(expiry_grid)), key=lambda idx: abs(expiry_grid[idx] - point.expiry))
        j = min(range(len(strike_grid)), key=lambda idx: abs(strike_grid[idx] - point.strike))
        vol_matrix[i][j] = point.volatility

    return VolatilitySurfaceGrid(
        strikes=strike_grid,
        expiries=expiry_grid,
        volatilities=vol_matrix,
    )
