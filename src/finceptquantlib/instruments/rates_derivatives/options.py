"""
Interest Rate Options

Implements:
- Caps: Series of call options on forward rates (caplets)
- Floors: Series of put options on forward rates (floorlets)
- Collars: Long cap + short floor
- Swaptions: Options on interest rate swaps

Pricing Models:
- Black-76 model for caplets/floorlets
- Bachelier (normal) model
- SABR volatility model support
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Union, Optional, List
import math
from scipy.stats import norm
from scipy.optimize import brentq

from finceptquantlib.core.foundations.types import OptionType, SwaptionType, VolatilityType, Frequency
from finceptquantlib.curves.curve import Curve
from finceptquantlib.scheduling.schedule import Schedule
from finceptquantlib.scheduling.calendar import Calendar
from finceptquantlib.scheduling.daycount import DayCountConvention, _get_convention


@dataclass
class OptionResult:
    """Result of option valuation."""
    premium: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    implied_vol: Optional[float] = None


@dataclass
class CapFloorResult:
    """Result of cap/floor valuation."""
    premium: float
    intrinsic_value: float
    time_value: float
    delta: float
    gamma: float
    vega: float
    caplet_values: List[float] = field(default_factory=list)
    forward_rates: List[float] = field(default_factory=list)


def black76_call(
    forward: float,
    strike: float,
    volatility: float,
    time_to_expiry: float,
    discount_factor: float,
) -> float:
    """
    Black-76 call option price.

    C = DF * (F * N(d1) - K * N(d2))

    where:
    d1 = (ln(F/K) + 0.5 * sigma^2 * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    """
    if time_to_expiry <= 0:
        return max(forward - strike, 0) * discount_factor

    if volatility <= 0:
        return max(forward - strike, 0) * discount_factor

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (math.log(forward / strike) + 0.5 * volatility ** 2 * time_to_expiry) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    return discount_factor * (forward * norm.cdf(d1) - strike * norm.cdf(d2))


def black76_put(
    forward: float,
    strike: float,
    volatility: float,
    time_to_expiry: float,
    discount_factor: float,
) -> float:
    """Black-76 put option price."""
    if time_to_expiry <= 0:
        return max(strike - forward, 0) * discount_factor

    if volatility <= 0:
        return max(strike - forward, 0) * discount_factor

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (math.log(forward / strike) + 0.5 * volatility ** 2 * time_to_expiry) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    return discount_factor * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))


def bachelier_call(
    forward: float,
    strike: float,
    volatility: float,
    time_to_expiry: float,
    discount_factor: float,
) -> float:
    """
    Bachelier (normal) model call option price.

    C = DF * ((F - K) * N(d) + sigma * sqrt(T) * n(d))

    where d = (F - K) / (sigma * sqrt(T))
    """
    if time_to_expiry <= 0:
        return max(forward - strike, 0) * discount_factor

    if volatility <= 0:
        return max(forward - strike, 0) * discount_factor

    sqrt_t = math.sqrt(time_to_expiry)
    d = (forward - strike) / (volatility * sqrt_t)

    return discount_factor * ((forward - strike) * norm.cdf(d) + volatility * sqrt_t * norm.pdf(d))


def bachelier_put(
    forward: float,
    strike: float,
    volatility: float,
    time_to_expiry: float,
    discount_factor: float,
) -> float:
    """Bachelier (normal) model put option price."""
    if time_to_expiry <= 0:
        return max(strike - forward, 0) * discount_factor

    if volatility <= 0:
        return max(strike - forward, 0) * discount_factor

    sqrt_t = math.sqrt(time_to_expiry)
    d = (forward - strike) / (volatility * sqrt_t)

    return discount_factor * ((strike - forward) * norm.cdf(-d) + volatility * sqrt_t * norm.pdf(-d))


def black76_greeks(
    forward: float,
    strike: float,
    volatility: float,
    time_to_expiry: float,
    discount_factor: float,
    option_type: OptionType = OptionType.CALL,
) -> dict:
    """
    Calculate Black-76 Greeks.

    Returns:
        Dictionary with delta, gamma, vega, theta
    """
    if time_to_expiry <= 0 or volatility <= 0:
        intrinsic = (forward - strike if option_type == OptionType.CALL
                    else strike - forward)
        return {
            "delta": 1.0 if intrinsic > 0 else 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
        }

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (math.log(forward / strike) + 0.5 * volatility ** 2 * time_to_expiry) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    if option_type == OptionType.CALL:
        delta = discount_factor * norm.cdf(d1)
    else:
        delta = -discount_factor * norm.cdf(-d1)

    gamma = discount_factor * norm.pdf(d1) / (forward * volatility * sqrt_t)
    vega = discount_factor * forward * norm.pdf(d1) * sqrt_t / 100  # Per 1% vol change

    # Theta (time decay per day): V = DF * Black(F, T), and with DF = e^{-rT} the passing
    # of time adds r * V on top of the volatility decay
    theta_term1 = -discount_factor * forward * norm.pdf(d1) * volatility / (2 * sqrt_t)
    rate = -math.log(discount_factor) / time_to_expiry
    if option_type == OptionType.CALL:
        undiscounted = forward * norm.cdf(d1) - strike * norm.cdf(d2)
    else:
        undiscounted = strike * norm.cdf(-d2) - forward * norm.cdf(-d1)
    theta = (theta_term1 + rate * discount_factor * undiscounted) / 365

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
    }


def implied_vol_black76(
    market_price: float,
    forward: float,
    strike: float,
    time_to_expiry: float,
    discount_factor: float,
    option_type: OptionType = OptionType.CALL,
    initial_guess: float = 0.2,
) -> float:
    """
    Calculate implied volatility using Black-76 model.

    Delegates to finceptquantlib.pricing.analytical.black76.black76_implied_vol.
    """
    if time_to_expiry <= 0:
        return 0.0

    from finceptquantlib.pricing.analytical.black76 import black76_implied_vol
    # Convert discount_factor to rate: df = exp(-r*T) => r = -ln(df)/T
    r = -math.log(discount_factor) / time_to_expiry if time_to_expiry > 0 else 0.0

    try:
        return black76_implied_vol(market_price, forward, strike, r, time_to_expiry, option_type)
    except ValueError:
        return initial_guess


class Caplet:
    """
    Single caplet (call option on forward rate).

    Payoff at payment date: max(R - K, 0) * dcf * Notional
    where R is the floating rate and K is the strike.
    """

    def __init__(
        self,
        notional: float,
        strike: float,
        fixing_date: date,
        payment_date: date,
        accrual_start: date,
        accrual_end: date,
        day_count: DayCountConvention,
    ):
        self.notional = notional
        self.strike = strike
        self.fixing_date = fixing_date
        self.payment_date = payment_date
        self.accrual_start = accrual_start
        self.accrual_end = accrual_end
        self.day_count = day_count
        self._dcf = day_count.year_fraction(accrual_start, accrual_end)

    def forward_rate(self, curve: Curve) -> float:
        """Get forward rate from curve."""
        t1 = curve.time_to_maturity(self.accrual_start)
        t2 = curve.time_to_maturity(self.accrual_end)
        if t1 <= 0:
            t1 = 0.001
        if t2 <= t1:
            t2 = t1 + self._dcf
        return curve.forward_rate(t1, t2)

    def price(
        self,
        valuation_date: date,
        curve: Curve,
        volatility: float,
        vol_type: VolatilityType = VolatilityType.LOGNORMAL,
    ) -> float:
        """
        Price the caplet.

        Args:
            valuation_date: Valuation date
            curve: Yield curve
            volatility: Cap volatility
            vol_type: Lognormal or normal

        Returns:
            Caplet premium
        """
        if self.payment_date <= valuation_date:
            return 0.0

        forward = self.forward_rate(curve)
        time_to_expiry = curve.time_to_maturity(self.fixing_date)
        df = curve.discount_factor(self.payment_date)

        if time_to_expiry <= 0:
            # Already fixed
            payoff = max(forward - self.strike, 0) * self._dcf * self.notional
            return payoff * df

        if vol_type == VolatilityType.LOGNORMAL:
            option_price = black76_call(forward, self.strike, volatility, time_to_expiry, 1.0)
        else:
            option_price = bachelier_call(forward, self.strike, volatility, time_to_expiry, 1.0)

        return option_price * self._dcf * self.notional * df


class Floorlet:
    """Single floorlet (put option on forward rate)."""

    def __init__(
        self,
        notional: float,
        strike: float,
        fixing_date: date,
        payment_date: date,
        accrual_start: date,
        accrual_end: date,
        day_count: DayCountConvention,
    ):
        self.notional = notional
        self.strike = strike
        self.fixing_date = fixing_date
        self.payment_date = payment_date
        self.accrual_start = accrual_start
        self.accrual_end = accrual_end
        self.day_count = day_count
        self._dcf = day_count.year_fraction(accrual_start, accrual_end)

    def forward_rate(self, curve: Curve) -> float:
        """Get forward rate from curve."""
        t1 = curve.time_to_maturity(self.accrual_start)
        t2 = curve.time_to_maturity(self.accrual_end)
        if t1 <= 0:
            t1 = 0.001
        if t2 <= t1:
            t2 = t1 + self._dcf
        return curve.forward_rate(t1, t2)

    def price(
        self,
        valuation_date: date,
        curve: Curve,
        volatility: float,
        vol_type: VolatilityType = VolatilityType.LOGNORMAL,
    ) -> float:
        """Price the floorlet."""
        if self.payment_date <= valuation_date:
            return 0.0

        forward = self.forward_rate(curve)
        time_to_expiry = curve.time_to_maturity(self.fixing_date)
        df = curve.discount_factor(self.payment_date)

        if time_to_expiry <= 0:
            payoff = max(self.strike - forward, 0) * self._dcf * self.notional
            return payoff * df

        if vol_type == VolatilityType.LOGNORMAL:
            option_price = black76_put(forward, self.strike, volatility, time_to_expiry, 1.0)
        else:
            option_price = bachelier_put(forward, self.strike, volatility, time_to_expiry, 1.0)

        return option_price * self._dcf * self.notional * df


class Cap:
    """
    Interest Rate Cap.

    A cap is a series of caplets (call options on forward rates).
    Provides protection against rising interest rates.

    The buyer of a cap receives payment when the floating rate
    exceeds the strike (cap rate).
    """

    def __init__(
        self,
        notional: float,
        strike: float,
        effective_date: date,
        termination_date: date,
        frequency: Union[str, Frequency] = Frequency.QUARTERLY,
        day_count: Union[str, DayCountConvention] = "ACT/360",
        calendar: Optional[Calendar] = None,
        fixing_days: int = 2,
    ):
        """
        Initialize cap.

        Args:
            notional: Notional principal
            strike: Cap strike rate
            start_date: First accrual start date
            end_date: Last accrual end date
            frequency: Payment frequency
            day_count: Day count convention
            calendar: Business day calendar
            fixing_days: Days before period start for fixing
        """
        self.notional = notional
        self.strike = strike
        self.start_date = effective_date
        self.end_date = termination_date

        if isinstance(frequency, str):
            frequency = Frequency[frequency.upper()]
        self.frequency = frequency

        if isinstance(day_count, str):
            day_count = _get_convention(day_count)
        self.day_count = day_count

        self.calendar = calendar
        self.fixing_days = fixing_days

        # Generate caplets
        self._schedule = Schedule(
            effective_date=effective_date,
            termination_date=termination_date,
            frequency=frequency,
            calendar=calendar,
        )
        self._caplets = self._create_caplets()

    def _create_caplets(self) -> List[Caplet]:
        """Create caplets from schedule."""
        caplets = []
        for period in self._schedule.periods:
            fixing_date = period.accrual_start - timedelta(days=self.fixing_days)
            payment_date = period.payment_date

            caplet = Caplet(
                notional=self.notional,
                strike=self.strike,
                fixing_date=fixing_date,
                payment_date=payment_date,
                accrual_start=period.accrual_start,
                accrual_end=period.accrual_end,
                day_count=self.day_count,
            )
            caplets.append(caplet)

        return caplets

    @property
    def caplets(self) -> List[Caplet]:
        """List of caplets."""
        return self._caplets

    def price(
        self,
        valuation_date: date,
        curve: Curve,
        volatility: Union[float, List[float]],
        vol_type: VolatilityType = VolatilityType.LOGNORMAL,
    ) -> float:
        """
        Price the cap.

        Args:
            valuation_date: Valuation date
            curve: Yield curve
            volatility: Flat vol or list of caplet vols
            vol_type: Volatility type

        Returns:
            Cap premium
        """
        if isinstance(volatility, (int, float)):
            vols = [volatility] * len(self._caplets)
        else:
            vols = volatility

        total = 0.0
        for caplet, vol in zip(self._caplets, vols):
            total += caplet.price(valuation_date, curve, vol, vol_type)

        return total

    def implied_volatility(
        self,
        market_price: float,
        valuation_date: date,
        curve: Curve,
        vol_type: VolatilityType = VolatilityType.LOGNORMAL,
        initial_guess: float = 0.2,
    ) -> float:
        """
        Calculate flat implied volatility from market price.

        Args:
            market_price: Observed cap price
            valuation_date: Valuation date
            curve: Yield curve
            vol_type: Volatility type
            initial_guess: Initial volatility guess

        Returns:
            Flat implied volatility
        """
        def objective(vol):
            return self.price(valuation_date, curve, vol, vol_type) - market_price

        try:
            return brentq(objective, 0.001, 3.0)
        except ValueError:
            return initial_guess

    def value(
        self,
        valuation_date: date,
        curve: Curve,
        volatility: float,
        vol_type: VolatilityType = VolatilityType.LOGNORMAL,
    ) -> CapFloorResult:
        """Full valuation with Greeks."""
        premium = self.price(valuation_date, curve, volatility, vol_type)

        # Calculate Greeks by bumping the curve 1bp
        # Delta (sensitivity to forward rates)
        curve_up = curve.shift(1)
        price_up = self.price(valuation_date, curve_up, volatility, vol_type)
        delta = price_up - premium

        curve_down = curve.shift(-1)
        price_down = self.price(valuation_date, curve_down, volatility, vol_type)
        gamma = (price_up - 2 * premium + price_down) / 1

        # Vega
        price_vol_up = self.price(valuation_date, curve, volatility + 0.01, vol_type)
        vega = price_vol_up - premium

        # Individual caplet values
        caplet_values = [c.price(valuation_date, curve, volatility, vol_type)
                        for c in self._caplets]
        forward_rates = [c.forward_rate(curve) for c in self._caplets]

        # Intrinsic value
        intrinsic = sum(
            max(fwd - self.strike, 0) * c._dcf * self.notional *
            curve.discount_factor(c.payment_date)
            for c, fwd in zip(self._caplets, forward_rates)
            if c.payment_date > valuation_date
        )

        return CapFloorResult(
            premium=premium,
            intrinsic_value=intrinsic,
            time_value=premium - intrinsic,
            delta=delta,
            gamma=gamma,
            vega=vega,
            caplet_values=caplet_values,
            forward_rates=forward_rates,
        )

    def __repr__(self) -> str:
        return (f"Cap(notional={self.notional:,.0f}, "
                f"strike={self.strike:.4%}, "
                f"{self.start_date} to {self.end_date})")


class Floor:
    """
    Interest Rate Floor.

    A floor is a series of floorlets (put options on forward rates).
    Provides protection against falling interest rates.
    """

    def __init__(
        self,
        notional: float,
        strike: float,
        effective_date: date,
        termination_date: date,
        frequency: Union[str, Frequency] = Frequency.QUARTERLY,
        day_count: Union[str, DayCountConvention] = "ACT/360",
        calendar: Optional[Calendar] = None,
        fixing_days: int = 2,
    ):
        """Initialize floor."""
        self.notional = notional
        self.strike = strike
        self.start_date = effective_date
        self.end_date = termination_date

        if isinstance(frequency, str):
            frequency = Frequency[frequency.upper()]
        self.frequency = frequency

        if isinstance(day_count, str):
            day_count = _get_convention(day_count)
        self.day_count = day_count

        self.calendar = calendar
        self.fixing_days = fixing_days

        self._schedule = Schedule(
            effective_date=effective_date,
            termination_date=termination_date,
            frequency=frequency,
            calendar=calendar,
        )
        self._floorlets = self._create_floorlets()

    def _create_floorlets(self) -> List[Floorlet]:
        """Create floorlets from schedule."""
        floorlets = []
        for period in self._schedule.periods:
            fixing_date = period.accrual_start - timedelta(days=self.fixing_days)
            payment_date = period.payment_date

            floorlet = Floorlet(
                notional=self.notional,
                strike=self.strike,
                fixing_date=fixing_date,
                payment_date=payment_date,
                accrual_start=period.accrual_start,
                accrual_end=period.accrual_end,
                day_count=self.day_count,
            )
            floorlets.append(floorlet)

        return floorlets

    @property
    def floorlets(self) -> List[Floorlet]:
        """List of floorlets."""
        return self._floorlets

    def price(
        self,
        valuation_date: date,
        curve: Curve,
        volatility: Union[float, List[float]],
        vol_type: VolatilityType = VolatilityType.LOGNORMAL,
    ) -> float:
        """Price the floor."""
        if isinstance(volatility, (int, float)):
            vols = [volatility] * len(self._floorlets)
        else:
            vols = volatility

        total = 0.0
        for floorlet, vol in zip(self._floorlets, vols):
            total += floorlet.price(valuation_date, curve, vol, vol_type)

        return total

    def implied_volatility(
        self,
        market_price: float,
        valuation_date: date,
        curve: Curve,
        vol_type: VolatilityType = VolatilityType.LOGNORMAL,
    ) -> float:
        """Calculate flat implied volatility."""
        def objective(vol):
            return self.price(valuation_date, curve, vol, vol_type) - market_price

        try:
            return brentq(objective, 0.001, 3.0)
        except ValueError:
            return 0.2

    def value(
        self,
        valuation_date: date,
        curve: Curve,
        volatility: float,
        vol_type: VolatilityType = VolatilityType.LOGNORMAL,
    ) -> CapFloorResult:
        """Full valuation."""
        premium = self.price(valuation_date, curve, volatility, vol_type)

        curve_up = curve.shift(1)
        price_up = self.price(valuation_date, curve_up, volatility, vol_type)
        delta = price_up - premium

        curve_down = curve.shift(-1)
        price_down = self.price(valuation_date, curve_down, volatility, vol_type)
        gamma = (price_up - 2 * premium + price_down)

        price_vol_up = self.price(valuation_date, curve, volatility + 0.01, vol_type)
        vega = price_vol_up - premium

        floorlet_values = [f.price(valuation_date, curve, volatility, vol_type)
                          for f in self._floorlets]
        forward_rates = [f.forward_rate(curve) for f in self._floorlets]

        intrinsic = sum(
            max(self.strike - fwd, 0) * f._dcf * self.notional *
            curve.discount_factor(f.payment_date)
            for f, fwd in zip(self._floorlets, forward_rates)
            if f.payment_date > valuation_date
        )

        return CapFloorResult(
            premium=premium,
            intrinsic_value=intrinsic,
            time_value=premium - intrinsic,
            delta=delta,
            gamma=gamma,
            vega=vega,
            caplet_values=floorlet_values,
            forward_rates=forward_rates,
        )

    def __repr__(self) -> str:
        return (f"Floor(notional={self.notional:,.0f}, "
                f"strike={self.strike:.4%}, "
                f"{self.start_date} to {self.end_date})")


class Collar:
    """
    Interest Rate Collar.

    Combination of long cap and short floor (or vice versa).
    Creates a bounded range for floating rate exposure.
    """

    def __init__(
        self,
        notional: float,
        cap_strike: float,
        floor_strike: float,
        effective_date: date,
        termination_date: date,
        frequency: Union[str, Frequency] = Frequency.QUARTERLY,
        day_count: Union[str, DayCountConvention] = "ACT/360",
        calendar: Optional[Calendar] = None,
        is_long_cap: bool = True,
    ):
        """
        Initialize collar.

        Args:
            notional: Notional principal
            cap_strike: Cap strike rate
            floor_strike: Floor strike rate
            start_date: Start date
            end_date: End date
            frequency: Payment frequency
            day_count: Day count convention
            calendar: Business day calendar
            is_long_cap: True = long cap/short floor
        """
        self.notional = notional
        self.cap_strike = cap_strike
        self.floor_strike = floor_strike
        self.is_long_cap = is_long_cap

        self.cap = Cap(
            notional=notional,
            strike=cap_strike,
            effective_date=effective_date,
            termination_date=termination_date,
            frequency=frequency,
            day_count=day_count,
            calendar=calendar,
        )

        self.floor = Floor(
            notional=notional,
            strike=floor_strike,
            effective_date=effective_date,
            termination_date=termination_date,
            frequency=frequency,
            day_count=day_count,
            calendar=calendar,
        )

    def price(
        self,
        valuation_date: date,
        curve: Curve,
        cap_vol: float,
        floor_vol: float,
        vol_type: VolatilityType = VolatilityType.LOGNORMAL,
    ) -> float:
        """
        Price the collar.

        Returns:
            Net collar premium (can be positive or negative)
        """
        cap_price = self.cap.price(valuation_date, curve, cap_vol, vol_type)
        floor_price = self.floor.price(valuation_date, curve, floor_vol, vol_type)

        if self.is_long_cap:
            return cap_price - floor_price
        else:
            return floor_price - cap_price

    def zero_cost_strike(
        self,
        valuation_date: date,
        curve: Curve,
        fixed_strike: float,
        volatility: float,
        find_cap: bool = True,
        vol_type: VolatilityType = VolatilityType.LOGNORMAL,
    ) -> float:
        """
        Find strike that makes collar zero-cost.

        Args:
            valuation_date: Valuation date
            curve: Yield curve
            fixed_strike: Known strike (cap or floor)
            volatility: Volatility for pricing
            find_cap: True = find cap strike, False = find floor strike

        Returns:
            Zero-cost strike
        """
        if find_cap:
            # Fix floor, find cap
            floor_price = Floor(
                self.notional, fixed_strike, self.cap.start_date, self.cap.end_date,
                self.cap.frequency, self.cap.day_count, self.cap.calendar
            ).price(valuation_date, curve, volatility, vol_type)

            def objective(cap_strike):
                cap = Cap(
                    self.notional, cap_strike, self.cap.start_date, self.cap.end_date,
                    self.cap.frequency, self.cap.day_count, self.cap.calendar
                )
                return cap.price(valuation_date, curve, volatility, vol_type) - floor_price

            return brentq(objective, fixed_strike, fixed_strike + 0.10)
        else:
            # Fix cap, find floor
            cap_price = Cap(
                self.notional, fixed_strike, self.cap.start_date, self.cap.end_date,
                self.cap.frequency, self.cap.day_count, self.cap.calendar
            ).price(valuation_date, curve, volatility, vol_type)

            def objective(floor_strike):
                floor = Floor(
                    self.notional, floor_strike, self.cap.start_date, self.cap.end_date,
                    self.cap.frequency, self.cap.day_count, self.cap.calendar
                )
                return floor.price(valuation_date, curve, volatility, vol_type) - cap_price

            return brentq(objective, fixed_strike - 0.10, fixed_strike)


class Swaption:
    """
    European Swaption.

    An option to enter into an interest rate swap at a future date.

    - Payer swaption: Right to pay fixed, receive floating
    - Receiver swaption: Right to receive fixed, pay floating

    Settlement can be:
    - Physical: Enter into the underlying swap
    - Cash: Receive swap value in cash
    """

    def __init__(
        self,
        notional: float,
        strike: float,
        expiry_date: date,
        swap_tenor_years: float,
        swaption_type: Union[str, SwaptionType] = SwaptionType.PAYER,
        fixed_frequency: Union[str, Frequency] = Frequency.SEMI_ANNUAL,
        float_frequency: Union[str, Frequency] = Frequency.QUARTERLY,
        fixed_day_count: Union[str, DayCountConvention] = "30/360",
        float_day_count: Union[str, DayCountConvention] = "ACT/360",
        calendar: Optional[Calendar] = None,
        settlement: str = "physical",
    ):
        """
        Initialize swaption.

        Args:
            notional: Notional principal
            strike: Strike swap rate
            expiry_date: Option expiry date
            swap_tenor_years: Underlying swap tenor in years
            swaption_type: PAYER or RECEIVER
            fixed_frequency: Fixed leg frequency
            float_frequency: Floating leg frequency
            fixed_day_count: Fixed leg day count
            float_day_count: Floating leg day count
            calendar: Business day calendar
            settlement: "physical" or "cash"
        """
        self.notional = notional
        self.strike = strike
        self.expiry_date = expiry_date
        self.swap_tenor_years = swap_tenor_years

        if isinstance(swaption_type, str):
            swaption_type = SwaptionType[swaption_type.upper()]
        self.swaption_type = swaption_type

        if isinstance(fixed_frequency, str):
            fixed_frequency = Frequency[fixed_frequency.upper()]
        self.fixed_frequency = fixed_frequency

        if isinstance(float_frequency, str):
            float_frequency = Frequency[float_frequency.upper()]
        self.float_frequency = float_frequency

        if isinstance(fixed_day_count, str):
            fixed_day_count = _get_convention(fixed_day_count)
        self.fixed_day_count = fixed_day_count

        if isinstance(float_day_count, str):
            float_day_count = _get_convention(float_day_count)
        self.float_day_count = float_day_count

        self.calendar = calendar
        self.settlement = settlement

    @property
    def swap_start_date(self) -> date:
        """Start date of underlying swap."""
        return self.expiry_date

    @property
    def swap_end_date(self) -> date:
        """End date of underlying swap."""
        return self.expiry_date + timedelta(days=int(self.swap_tenor_years * 365))

    def forward_swap_rate(self, curve: Curve) -> float:
        """
        Calculate forward swap rate.

        The forward swap rate is the rate that makes the forward swap have zero NPV.
        """
        # Calculate floating leg PV and annuity
        schedule = Schedule(
            effective_date=self.swap_start_date,
            termination_date=self.swap_end_date,
            frequency=self.fixed_frequency,
            calendar=self.calendar,
        )

        annuity = 0.0
        for start, end in schedule.periods:
            dcf = self.fixed_day_count.year_fraction(start, end)
            df = curve.discount_factor(end)
            annuity += dcf * df

        if annuity == 0:
            return 0.0

        # Float leg = 1 - DF(end)
        df_start = curve.discount_factor(self.swap_start_date)
        df_end = curve.discount_factor(self.swap_end_date)
        float_pv = df_start - df_end

        return float_pv / annuity

    def annuity(self, curve: Curve) -> float:
        """Calculate swap annuity (present value of basis point)."""
        schedule = Schedule(
            effective_date=self.swap_start_date,
            termination_date=self.swap_end_date,
            frequency=self.fixed_frequency,
            calendar=self.calendar,
        )

        total = 0.0
        for start, end in schedule.periods:
            dcf = self.fixed_day_count.year_fraction(start, end)
            df = curve.discount_factor(end)
            total += dcf * df

        return self.notional * total

    def price(
        self,
        valuation_date: date,
        curve: Curve,
        volatility: float,
        vol_type: VolatilityType = VolatilityType.LOGNORMAL,
    ) -> float:
        """
        Price the swaption using Black's model.

        Args:
            valuation_date: Valuation date
            curve: Yield curve
            volatility: Swaption volatility
            vol_type: Lognormal or normal

        Returns:
            Swaption premium
        """
        if self.expiry_date <= valuation_date:
            return 0.0

        forward = self.forward_swap_rate(curve)
        time_to_expiry = curve.time_to_maturity(self.expiry_date)
        annuity = self.annuity(curve)

        if vol_type == VolatilityType.LOGNORMAL:
            if self.swaption_type == SwaptionType.PAYER:
                option_value = black76_call(forward, self.strike, volatility, time_to_expiry, 1.0)
            else:
                option_value = black76_put(forward, self.strike, volatility, time_to_expiry, 1.0)
        else:
            if self.swaption_type == SwaptionType.PAYER:
                option_value = bachelier_call(forward, self.strike, volatility, time_to_expiry, 1.0)
            else:
                option_value = bachelier_put(forward, self.strike, volatility, time_to_expiry, 1.0)

        return option_value * annuity

    def implied_volatility(
        self,
        market_price: float,
        valuation_date: date,
        curve: Curve,
        vol_type: VolatilityType = VolatilityType.LOGNORMAL,
    ) -> float:
        """Calculate implied volatility from market price."""
        def objective(vol):
            return self.price(valuation_date, curve, vol, vol_type) - market_price

        try:
            return brentq(objective, 0.001, 3.0)
        except ValueError:
            return 0.2

    def greeks(
        self,
        valuation_date: date,
        curve: Curve,
        volatility: float,
        vol_type: VolatilityType = VolatilityType.LOGNORMAL,
    ) -> OptionResult:
        """
        Calculate swaption Greeks.

        Returns:
            OptionResult with delta, gamma, vega, theta
        """
        base_price = self.price(valuation_date, curve, volatility, vol_type)

        # Delta (rate sensitivity)
        curve_up = curve.shift(1)
        price_up = self.price(valuation_date, curve_up, volatility, vol_type)
        delta = price_up - base_price

        curve_down = curve.shift(-1)
        price_down = self.price(valuation_date, curve_down, volatility, vol_type)
        gamma = (price_up - 2 * base_price + price_down)

        # Vega
        price_vol_up = self.price(valuation_date, curve, volatility + 0.01, vol_type)
        vega = price_vol_up - base_price

        # Theta (1-day time decay)
        next_date = valuation_date + timedelta(days=1)
        price_next = self.price(next_date, curve, volatility, vol_type)
        theta = price_next - base_price

        # Rho (discount rate sensitivity)
        # Approximated by DV01 of annuity
        rho = self.annuity(curve) / 10000

        return OptionResult(
            premium=base_price,
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho=rho,
            implied_vol=volatility,
        )

    def __repr__(self) -> str:
        return (f"Swaption({self.swaption_type.value}, "
                f"notional={self.notional:,.0f}, "
                f"strike={self.strike:.4%}, "
                f"expiry={self.expiry_date}, "
                f"tenor={self.swap_tenor_years}Y)")
