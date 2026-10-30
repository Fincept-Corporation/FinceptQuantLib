"""
FX Forward Engine

Calculate FX forward rates from interest rate differentials.

Features:
- Forward points calculation
- Outright forward rates
- Broken date interpolation
- Cross-currency basis adjustment
- NDF pricing
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

from finceptquantlib.core.foundations.types import (
    Currency, Tenor, as_tenor
)
from .rates import parse_pair, get_pip_size

if TYPE_CHECKING:
    from finceptquantlib.curves.curve import Curve


# =============================================================================
# Forward Quote
# =============================================================================

@dataclass
class FXForwardQuote:
    """
    FX forward quote.

    Can be expressed as outright rate or forward points.

    Attributes:
        pair: Currency pair
        spot: Spot rate
        forward_points: Forward points (in pips or absolute)
        outright: Outright forward rate
        value_date: Forward value date
        spot_date: Spot value date
    """
    pair: str
    spot: float
    value_date: date
    spot_date: Optional[date] = None
    forward_points: Optional[float] = None
    outright: Optional[float] = None
    points_in_pips: bool = True

    def __post_init__(self) -> None:
        # Calculate missing values
        if self.outright is None and self.forward_points is not None:
            if self.points_in_pips:
                pip_size = get_pip_size(self.pair)
                self.outright = self.spot + self.forward_points * pip_size
            else:
                self.outright = self.spot + self.forward_points

        elif self.forward_points is None and self.outright is not None:
            diff = self.outright - self.spot
            if self.points_in_pips:
                pip_size = get_pip_size(self.pair)
                self.forward_points = diff / pip_size
            else:
                self.forward_points = diff

    @property
    def premium_discount(self) -> str:
        """Whether forward is at premium or discount."""
        if self.forward_points is None:
            return "unknown"
        return "premium" if self.forward_points > 0 else "discount"

    @property
    def annualized_basis(self) -> Optional[float]:
        """Annualized forward basis in percentage."""
        if self.outright is None or self.spot_date is None:
            return None
        days = (self.value_date - self.spot_date).days
        if days <= 0:
            return None
        return (self.outright / self.spot - 1) * 365 / days * 100

    def __repr__(self) -> str:
        pts = f"{self.forward_points:+.2f}" if self.forward_points else "N/A"
        return f"FXForwardQuote({self.pair} {self.value_date}: {pts} pts, {self.outright:.5f})"


# =============================================================================
# FX Forward Calculator
# =============================================================================

class FXForwardCalculator:
    """
    Calculate FX forwards from interest rate curves.

    Uses covered interest rate parity:
    F = S * (1 + r_d * t) / (1 + r_f * t)

    Or with continuous compounding:
    F = S * exp((r_d - r_f) * t)

    Where:
    - F = Forward rate
    - S = Spot rate
    - r_d = Domestic (quote currency) rate
    - r_f = Foreign (base currency) rate
    - t = Time to maturity

    Example:
        >>> calc = FXForwardCalculator(  # doctest: +SKIP
        ...     pair="EUR/USD",
        ...     spot=1.0850,
        ...     base_curve=eur_curve,
        ...     quote_curve=usd_curve
        ... )
        >>> calc.forward("6M")  # 6-month forward  # doctest: +SKIP
        >>> calc.points("1Y")   # 1-year forward points  # doctest: +SKIP
    """

    def __init__(
        self,
        pair: str,
        spot: float,
        base_curve: "Curve",
        quote_curve: "Curve",
        spot_date: Optional[date] = None,
        basis_curve: Optional["Curve"] = None,
        continuous: bool = False
    ) -> None:
        """
        Initialize forward calculator.

        Args:
            pair: Currency pair (e.g., "EUR/USD")
            spot: Spot FX rate
            base_curve: Interest rate curve for base currency
            quote_curve: Interest rate curve for quote currency
            spot_date: Spot value date (default: curve base date)
            basis_curve: Cross-currency basis curve (optional)
            continuous: Use continuous compounding
        """
        self.pair = pair.upper()
        self.spot = spot
        self.base_currency, self.quote_currency = parse_pair(pair)
        self.base_curve = base_curve
        self.quote_curve = quote_curve
        self.spot_date = spot_date or base_curve.reference_date
        self.basis_curve = basis_curve
        self.continuous = continuous

    def forward(
        self,
        tenor_or_date: Union[str, Tenor, date],
        include_basis: bool = True
    ) -> float:
        """
        Calculate outright forward rate.

        Args:
            tenor_or_date: Tenor string, Tenor object, or target date
            include_basis: Include cross-currency basis adjustment

        Returns:
            Outright forward rate
        """
        target_date = self._resolve_date(tenor_or_date)

        # Calculate time to maturity
        t = self._year_fraction(self.spot_date, target_date)

        if t <= 0:
            return self.spot

        # Get discount factors or rates
        df_base = self.base_curve.discount_factor(target_date)
        df_quote = self.quote_curve.discount_factor(target_date)

        # Forward = Spot * DF_base / DF_quote
        # This follows from: F = S * (1 + r_q * t) / (1 + r_b * t)
        # Which is equivalent to: F = S * DF_b / DF_q when using proper DFs
        forward = self.spot * df_base / df_quote

        # Apply cross-currency basis if available
        if include_basis and self.basis_curve is not None:
            basis_spread = self.basis_curve.zero_rate(target_date)
            if self.continuous:
                forward *= math.exp(-basis_spread * t)
            else:
                forward *= 1 / (1 + basis_spread * t)

        return forward

    def points(
        self,
        tenor_or_date: Union[str, Tenor, date],
        in_pips: bool = True,
        include_basis: bool = True
    ) -> float:
        """
        Calculate forward points.

        Args:
            tenor_or_date: Tenor or target date
            in_pips: Return points in pips (vs absolute)
            include_basis: Include cross-currency basis

        Returns:
            Forward points
        """
        fwd = self.forward(tenor_or_date, include_basis)
        diff = fwd - self.spot

        if in_pips:
            pip_size = get_pip_size(self.pair)
            return diff / pip_size

        return diff

    def quote(
        self,
        tenor_or_date: Union[str, Tenor, date],
        include_basis: bool = True
    ) -> FXForwardQuote:
        """
        Get full forward quote.

        Args:
            tenor_or_date: Tenor or target date
            include_basis: Include cross-currency basis

        Returns:
            FXForwardQuote object
        """
        target_date = self._resolve_date(tenor_or_date)
        fwd = self.forward(target_date, include_basis)
        pts = self.points(target_date, in_pips=True, include_basis=include_basis)

        return FXForwardQuote(
            pair=self.pair,
            spot=self.spot,
            value_date=target_date,
            spot_date=self.spot_date,
            forward_points=pts,
            outright=fwd,
            points_in_pips=True
        )

    def forward_curve(
        self,
        tenors: List[Union[str, Tenor]],
        include_basis: bool = True
    ) -> Dict[date, float]:
        """
        Generate forward curve for multiple tenors.

        Returns:
            Dictionary of date -> forward rate
        """
        result = {}
        for tenor in tenors:
            target_date = self._resolve_date(tenor)
            result[target_date] = self.forward(target_date, include_basis)
        return result

    def implied_rate_differential(
        self,
        tenor_or_date: Union[str, Tenor, date]
    ) -> float:
        """
        Calculate implied interest rate differential from forward.

        Returns r_quote - r_base implied by the forward.
        """
        target_date = self._resolve_date(tenor_or_date)
        t = self._year_fraction(self.spot_date, target_date)

        if t <= 0:
            return 0.0

        fwd = self.forward(tenor_or_date, include_basis=False)

        if self.continuous:
            return math.log(fwd / self.spot) / t
        else:
            return (fwd / self.spot - 1) / t

    def swap_points(
        self,
        near_tenor: Union[str, Tenor, date],
        far_tenor: Union[str, Tenor, date]
    ) -> Tuple[float, float]:
        """
        Calculate FX swap points (near leg and far leg).

        Returns:
            (near_points, far_points) in pips
        """
        near_pts = self.points(near_tenor, in_pips=True)
        far_pts = self.points(far_tenor, in_pips=True)
        return near_pts, far_pts

    def _resolve_date(self, tenor_or_date: Union[str, Tenor, date]) -> date:
        """Resolve tenor or date to target date."""
        if isinstance(tenor_or_date, date):
            return tenor_or_date
        tenor = as_tenor(tenor_or_date)
        return tenor.add_to_date(self.spot_date)

    def _year_fraction(self, start: date, end: date) -> float:
        """Calculate year fraction (ACT/365)."""
        return (end - start).days / 365.0

    def __repr__(self) -> str:
        return f"FXForwardCalculator({self.pair}, spot={self.spot:.5f})"


# =============================================================================
# Convenience Functions
# =============================================================================

def calculate_forward_points(
    spot: float,
    base_rate: float,
    quote_rate: float,
    days: int,
    pair: str = "EUR/USD",
    in_pips: bool = True
) -> float:
    """
    Calculate forward points from interest rates.

    Simple formula without curves.

    Args:
        spot: Spot FX rate
        base_rate: Base currency interest rate (decimal)
        quote_rate: Quote currency interest rate (decimal)
        days: Days to forward date
        pair: Currency pair (for pip calculation)
        in_pips: Return in pips

    Returns:
        Forward points
    """
    t = days / 365.0
    forward = spot * (1 + quote_rate * t) / (1 + base_rate * t)
    diff = forward - spot

    if in_pips:
        pip_size = get_pip_size(pair)
        return diff / pip_size

    return diff


def calculate_outright_forward(
    spot: float,
    base_rate: float,
    quote_rate: float,
    days: int
) -> float:
    """
    Calculate outright forward rate from interest rates.

    Args:
        spot: Spot FX rate
        base_rate: Base currency rate (decimal)
        quote_rate: Quote currency rate (decimal)
        days: Days to forward date

    Returns:
        Outright forward rate
    """
    t = days / 365.0
    return spot * (1 + quote_rate * t) / (1 + base_rate * t)


def implied_deposit_rate(
    spot: float,
    forward: float,
    known_rate: float,
    days: int,
    is_base_rate: bool = True
) -> float:
    """
    Calculate implied deposit rate from forward.

    Args:
        spot: Spot rate
        forward: Forward rate
        known_rate: Known interest rate (decimal)
        days: Days to forward
        is_base_rate: If True, known_rate is base rate; calc quote rate

    Returns:
        Implied interest rate
    """
    t = days / 365.0

    if is_base_rate:
        # known_rate is r_b, solve for r_q
        # F = S * (1 + r_q * t) / (1 + r_b * t)
        # r_q = ((F/S) * (1 + r_b * t) - 1) / t
        return ((forward / spot) * (1 + known_rate * t) - 1) / t
    else:
        # known_rate is r_q, solve for r_b
        # r_b = ((S/F) * (1 + r_q * t) - 1) / t
        return ((spot / forward) * (1 + known_rate * t) - 1) / t


# =============================================================================
# NDF Pricing
# =============================================================================

@dataclass
class NDFQuote:
    """
    Non-Deliverable Forward quote.

    NDFs settle in a convertible currency (usually USD) based on
    the difference between contracted rate and fixing rate.

    Attributes:
        pair: Currency pair (e.g., "USD/KRW")
        notional: Notional amount
        notional_currency: Currency of notional
        contracted_rate: Agreed forward rate
        fixing_date: Date when reference rate is fixed
        settlement_date: Cash settlement date
        settlement_currency: Currency for settlement
    """
    pair: str
    notional: float
    notional_currency: Currency
    contracted_rate: float
    fixing_date: date
    settlement_date: date
    settlement_currency: Currency = Currency.USD

    def settlement_amount(self, fixing_rate: float) -> float:
        """
        Calculate settlement amount given fixing rate.

        For USD/XXX NDF with USD notional:
        Settlement = Notional * (1/contracted - 1/fixing) * fixing
                   = Notional * (fixing - contracted) / contracted

        Args:
            fixing_rate: Actual fixing rate at fixing_date

        Returns:
            Settlement amount in settlement currency
        """
        base, quote = parse_pair(self.pair)

        if self.notional_currency == base:
            # Notional in base currency (e.g., USD notional for USD/KRW)
            return self.notional * (fixing_rate - self.contracted_rate) / self.contracted_rate
        else:
            # Notional in quote currency
            return self.notional * (1 / self.contracted_rate - 1 / fixing_rate)

    def mark_to_market(
        self,
        current_forward: float,
        discount_factor: float
    ) -> float:
        """
        Calculate mark-to-market value.

        Args:
            current_forward: Current forward rate for fixing date
            discount_factor: DF from today to settlement date

        Returns:
            MTM value in settlement currency
        """
        expected_settlement = self.settlement_amount(current_forward)
        return expected_settlement * discount_factor

    def __repr__(self) -> str:
        return (
            f"NDFQuote({self.pair}, {self.notional:,.0f} {self.notional_currency.value}, "
            f"rate={self.contracted_rate:.4f}, fixing={self.fixing_date})"
        )
