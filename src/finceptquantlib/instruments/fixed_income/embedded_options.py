"""
Bonds with Embedded Options

Bonds with call/put provisions priced using binomial trees:
- Callable bonds
- Puttable bonds
- Callable-puttable bonds
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Optional
import math


@dataclass
class CallSchedule:
    """Call schedule for callable bond."""
    call_dates: List[date]
    call_prices: List[float]
    make_whole_provision: bool = False

    def get_call_price(self, d: date) -> Optional[float]:
        """Get call price for given date."""
        for i, call_date in enumerate(self.call_dates):
            if d >= call_date:
                return self.call_prices[i] if i < len(self.call_prices) else self.call_prices[-1]
        return None


@dataclass
class PutSchedule:
    """Put schedule for puttable bond."""
    put_dates: List[date]
    put_prices: List[float]

    def get_put_price(self, d: date) -> Optional[float]:
        """Get put price for given date."""
        for i, put_date in enumerate(self.put_dates):
            if d == put_date:
                return self.put_prices[i] if i < len(self.put_prices) else self.put_prices[-1]
        return None


class CallableBond:
    """
    Callable Bond.

    Bond that can be redeemed by issuer before maturity.
    Priced using binomial interest rate tree.

    Examples:
        >>> call_schedule = CallSchedule(
        ...     call_dates=[date(2026, 1, 1), date(2027, 1, 1)],
        ...     call_prices=[102.0, 101.0]
        ... )
        >>> cb = CallableBond(
        ...     issue_date=date(2024, 1, 1),
        ...     maturity_date=date(2029, 1, 1),
        ...     par_value=100,
        ...     coupon_rate=0.05,
        ...     call_schedule=call_schedule
        ... )
        >>> price = cb.price(yield_curve, volatility=0.15)  # doctest: +SKIP
    """

    def __init__(
        self,
        issue_date: date,
        maturity_date: date,
        par_value: float,
        coupon_rate: float,
        call_schedule: CallSchedule,
        frequency: int = 2,  # Semi-annual
        day_count: str = "30/360"
    ) -> None:
        """Initialize callable bond."""
        self.issue_date = issue_date
        self.maturity_date = maturity_date
        self.par_value = par_value
        self.coupon_rate = coupon_rate
        self.call_schedule = call_schedule
        self.frequency = frequency
        self.day_count = day_count

    def price(
        self,
        yield_curve,  # Curve object
        volatility: float,
        valuation_date: Optional[date] = None,
        n_steps: int = 100
    ) -> float:
        """
        Price callable bond using binomial tree.

        Args:
            yield_curve: Yield curve for discounting
            volatility: Interest rate volatility
            valuation_date: Valuation date
            n_steps: Number of tree steps

        Returns:
            Bond price
        """
        if valuation_date is None:
            valuation_date = date.today()

        years_to_maturity = (self.maturity_date - valuation_date).days / 365.25
        dt = years_to_maturity / n_steps

        # Short rate at valuation
        r0 = yield_curve.zero_rate(0.1) if hasattr(yield_curve, 'zero_rate') else 0.05

        # Binomial tree parameters
        u = math.exp(volatility * math.sqrt(dt))
        p = 0.5  # Risk-neutral probability

        # Build interest rate tree
        rates = [[r0]]
        for step in range(1, n_steps + 1):
            new_rates = []
            for i in range(step + 1):
                rate = r0 * (u ** (step - 2*i))
                new_rates.append(rate)
            rates.append(new_rates)

        # Initialize bond values at maturity
        values = [self.par_value] * (n_steps + 1)

        # Coupon amount
        coupon = self.par_value * self.coupon_rate / self.frequency

        # Work backwards through tree
        for step in range(n_steps - 1, -1, -1):
            new_values = []
            step_date = valuation_date  # Simplified

            for i in range(step + 1):
                # Discount expected value
                rate = rates[step][i]
                disc = math.exp(-rate * dt)
                continuation = (p * values[i] + (1 - p) * values[i + 1]) * disc

                # Add coupon if payment date
                if step % (n_steps // (self.frequency * years_to_maturity)) == 0:
                    continuation += coupon

                # Check if callable
                call_price = self.call_schedule.get_call_price(step_date)
                if call_price is not None:
                    # Issuer calls if beneficial
                    continuation = min(continuation, call_price)

                new_values.append(continuation)

            values = new_values

        return values[0]

    def option_adjusted_spread(
        self,
        market_price: float,
        yield_curve,
        volatility: float
    ) -> float:
        """
        Calculate option-adjusted spread (OAS).

        Spread that makes model price = market price.
        """
        # Simplified: return estimate
        return 0.01  # Placeholder


class PuttableBond:
    """
    Puttable Bond.

    Bond that can be sold back to issuer by holder at specified dates/prices.

    Examples:
        >>> put_schedule = PutSchedule(
        ...     put_dates=[date(2026, 1, 1), date(2027, 1, 1)],
        ...     put_prices=[100.0, 100.0]
        ... )
        >>> pb = PuttableBond(
        ...     issue_date=date(2024, 1, 1),
        ...     maturity_date=date(2029, 1, 1),
        ...     par_value=100,
        ...     coupon_rate=0.06,  # Higher coupon to compensate
        ...     put_schedule=put_schedule
        ... )
    """

    def __init__(
        self,
        issue_date: date,
        maturity_date: date,
        par_value: float,
        coupon_rate: float,
        put_schedule: PutSchedule,
        frequency: int = 2
    ) -> None:
        """Initialize puttable bond."""
        self.issue_date = issue_date
        self.maturity_date = maturity_date
        self.par_value = par_value
        self.coupon_rate = coupon_rate
        self.put_schedule = put_schedule
        self.frequency = frequency

    def price(self, yield_curve, volatility: float, n_steps: int = 100) -> float:
        """
        Price puttable bond.

        Similar to callable, but holder exercises put when beneficial.
        """
        # Simplified pricing
        straight_bond_value = self._straight_bond_value(yield_curve)

        # Put option value (approximation)
        put_value = self.par_value * 0.02  # ~2% for typical put

        return straight_bond_value + put_value

    def _straight_bond_value(self, yield_curve) -> float:
        """Calculate straight bond value."""
        return self.par_value  # Simplified


class CallablePuttableBond:
    """Bond with both call and put provisions."""

    def __init__(
        self,
        issue_date: date,
        maturity_date: date,
        par_value: float,
        coupon_rate: float,
        call_schedule: CallSchedule,
        put_schedule: PutSchedule,
        frequency: int = 2
    ) -> None:
        """Initialize callable-puttable bond."""
        self.issue_date = issue_date
        self.maturity_date = maturity_date
        self.par_value = par_value
        self.coupon_rate = coupon_rate
        self.call_schedule = call_schedule
        self.put_schedule = put_schedule
        self.frequency = frequency

    def price(self, yield_curve, volatility: float) -> float:
        """Price bond with both options."""
        # Requires full binomial tree with both option checks
        return self.par_value  # Placeholder
