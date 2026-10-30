"""
Multi-Curve Framework

Post-2008 market standard: separate curves for discounting and projection.
- OIS curve for discounting
- IBOR curves for projection
- Cross-currency basis curves
- FX-adjusted curves
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

from finceptquantlib.curves.curve import Curve, ZeroCurve
from finceptquantlib.core.foundations.types import Currency


@dataclass
class CurveSet:
    """
    Complete set of curves for multi-curve pricing.

    Attributes:
        discount_curve: OIS curve for discounting
        projection_curves: IBOR curves by tenor (3M, 6M, etc.)
        fx_curve: FX forward curve (if cross-currency)
        basis_curves: Basis spread curves
    """
    discount_curve: Curve
    projection_curves: Dict[str, Curve]  # Key: tenor like "3M", "6M"
    fx_curve: Optional[Curve] = None
    basis_curves: Optional[Dict[str, Curve]] = None


class MultiCurveFramework:
    """
    Multi-Curve Framework Manager.

    Manages consistent set of curves for post-crisis derivative pricing.

    Examples:
        >>> from datetime import date
        >>> from finceptquantlib.curves import ZeroCurve
        >>> ois = ZeroCurve(date(2024, 1, 15), [1, 2, 5, 10], [0.040, 0.041, 0.042, 0.043])
        >>> sofr_3m = ZeroCurve(date(2024, 1, 15), [1, 2, 5, 10], [0.042, 0.043, 0.044, 0.045])
        >>> mcf = MultiCurveFramework(currency=Currency.USD)
        >>> mcf.set_discount_curve(ois)
        >>> mcf.add_projection_curve("3M", sofr_3m)
        >>> round(mcf.forward_rate("3M", 1.0, 1.25), 6)
        0.04325
    """

    def __init__(self, currency: Currency) -> None:
        """
        Initialize multi-curve framework.

        Args:
            currency: Currency for curves
        """
        self.currency = currency
        self.discount_curve: Optional[Curve] = None
        self.projection_curves: Dict[str, Curve] = {}
        self.basis_curves: Dict[str, Curve] = {}

    def set_discount_curve(self, curve: Curve) -> None:
        """Set OIS discount curve."""
        self.discount_curve = curve

    def add_projection_curve(self, tenor: str, curve: Curve) -> None:
        """
        Add projection curve for specific tenor.

        Args:
            tenor: Index tenor (e.g., "3M", "6M")
            curve: Forward curve for that tenor
        """
        self.projection_curves[tenor] = curve

    def add_basis_curve(self, tenor_pair: str, curve: Curve) -> None:
        """
        Add basis spread curve.

        Args:
            tenor_pair: Tenor pair like "3Mv6M" (3M vs 6M basis)
            curve: Basis spread curve
        """
        self.basis_curves[tenor_pair] = curve

    def discount_factor(self, time: float) -> float:
        """
        Get discount factor from OIS curve.

        Args:
            time: Time in years

        Returns:
            Discount factor
        """
        if self.discount_curve is None:
            raise ValueError("Discount curve not set")

        return self.discount_curve.discount_factor(time)

    def forward_rate(
        self,
        tenor: str,
        start_time: float,
        end_time: float
    ) -> float:
        """
        Get forward rate for specific tenor.

        Args:
            tenor: Index tenor
            start_time: Forward start (years)
            end_time: Forward end (years)

        Returns:
            Forward rate
        """
        if tenor not in self.projection_curves:
            raise ValueError(f"No projection curve for tenor {tenor}")

        curve = self.projection_curves[tenor]
        return curve.forward_rate(start_time, end_time)

    def basis_spread(self, tenor_pair: str, time: float) -> float:
        """
        Get basis spread between two tenors.

        Args:
            tenor_pair: Tenor pair (e.g., "3Mv6M")
            time: Time point

        Returns:
            Basis spread
        """
        if tenor_pair not in self.basis_curves:
            return 0.0

        # Basis spread = difference in forward rates between the two projection curves
        parts = tenor_pair.upper().split("V")
        if len(parts) == 2 and parts[0] in self.projection_curves and parts[1] in self.projection_curves:
            dt = 0.25  # Quarter for forward rate estimation
            rate1 = self.projection_curves[parts[0]].forward_rate(time, time + dt)
            rate2 = self.projection_curves[parts[1]].forward_rate(time, time + dt)
            return rate1 - rate2

        # Fallback: read spread from explicit basis curve
        curve = self.basis_curves[tenor_pair]
        return curve.zero_rate(time) if hasattr(curve, 'zero_rate') else 0.0


class CrossCurrencyBasisCurve:
    """
    Cross-Currency Basis Curve.

    Represents basis spread between two currencies in cross-currency swaps.
    Post-2008: non-zero due to funding costs and regulatory constraints.

    Examples:
        >>> basis = CrossCurrencyBasisCurve(
        ...     domestic=Currency.USD,
        ...     foreign=Currency.EUR,
        ...     tenors=[1, 2, 3, 5, 10],
        ...     basis_spreads=[-0.0010, -0.0012, -0.0015, -0.0020, -0.0025]
        ... )
        >>> spread_5y = basis.basis_spread(5.0)
    """

    def __init__(
        self,
        domestic: Currency,
        foreign: Currency,
        tenors: List[float],
        basis_spreads: List[float],
        reference_date: Optional[date] = None
    ) -> None:
        """
        Initialize cross-currency basis curve.

        Args:
            domestic: Domestic currency
            foreign: Foreign currency
            tenors: Tenor points (years)
            basis_spreads: Basis spreads (negative = foreign pays more)
            reference_date: Curve reference date
        """
        self.domestic = domestic
        self.foreign = foreign
        self.tenors = tenors
        self.basis_spreads = basis_spreads
        self.reference_date = reference_date or date.today()

    def basis_spread(self, time: float) -> float:
        """
        Interpolate basis spread at given time.

        Args:
            time: Time in years

        Returns:
            Basis spread
        """
        # Linear interpolation
        if time <= self.tenors[0]:
            return self.basis_spreads[0]
        elif time >= self.tenors[-1]:
            return self.basis_spreads[-1]

        for i in range(len(self.tenors) - 1):
            if self.tenors[i] <= time <= self.tenors[i + 1]:
                # Interpolate
                t1, t2 = self.tenors[i], self.tenors[i + 1]
                s1, s2 = self.basis_spreads[i], self.basis_spreads[i + 1]
                return s1 + (s2 - s1) * (time - t1) / (t2 - t1)

        return 0.0


class OISDiscountCurve(ZeroCurve):
    """
    OIS (Overnight Index Swap) Discount Curve.

    Risk-free discount curve based on overnight rates (SOFR, ESTR, SONIA).
    Used as discounting curve in multi-curve framework. The rates are
    continuously compounded zero rates at each tenor.
    """

    def __init__(
        self,
        reference_date: date,
        tenors: List[float],
        ois_rates: List[float],
        overnight_index: str = "SOFR"
    ) -> None:
        """
        Initialize OIS curve.

        Args:
            reference_date: Curve date
            tenors: Tenor points
            ois_rates: OIS swap rates
            overnight_index: Overnight index name
        """
        super().__init__(reference_date, tenors, ois_rates)
        self.overnight_index = overnight_index


class IBORProjectionCurve(ZeroCurve):
    """
    IBOR Projection Curve.

    Projection curve for IBOR rates (LIBOR, EURIBOR, etc.) or their replacements,
    used for projecting floating leg cash flows. Like the OIS curve it is held as
    continuously compounded zero rates (a pseudo-discount curve); IBOR forwards are
    projected from it with ``forward_rate``.
    """

    def __init__(
        self,
        reference_date: date,
        tenors: List[float],
        zero_rates: List[float],
        index_tenor: str = "3M"
    ) -> None:
        """
        Initialize IBOR projection curve.

        Args:
            reference_date: Curve date
            tenors: Tenor points
            zero_rates: Zero rates of the projection curve
            index_tenor: Index tenor (3M, 6M, etc.)
        """
        super().__init__(reference_date, tenors, zero_rates)
        self.index_tenor = index_tenor


def calibrate_multicurve(
    ois_instruments: List,
    ibor_instruments: Dict[str, List],
    basis_instruments: Optional[List] = None
) -> MultiCurveFramework:
    """
    Bootstrap multi-curve framework from market instruments.

    Args:
        ois_instruments: OIS swaps for discount curve
        ibor_instruments: Dict of IBOR instruments by tenor
        basis_instruments: Basis swaps (optional)

    Returns:
        Calibrated MultiCurveFramework
    """
    # Simplified implementation
    # Full version would simultaneously bootstrap all curves
    raise NotImplementedError("Multi-curve bootstrapping not yet implemented")
