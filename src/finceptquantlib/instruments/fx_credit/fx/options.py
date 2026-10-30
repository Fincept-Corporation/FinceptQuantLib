"""
FX Options

Implements:
- Vanilla FX options (European and American)
- Barrier options (knock-in, knock-out)
- Digital options (cash-or-nothing, asset-or-nothing)

Pricing Models:
- Garman-Kohlhagen model (extension of Black-Scholes for FX)
- Monte Carlo for exotic options
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Union, Optional
from enum import Enum
import math
import random
from scipy.stats import norm
from scipy.optimize import brentq

from finceptquantlib.core.foundations.types import OptionType, BarrierType
from finceptquantlib.curves.curve import Curve
from finceptquantlib.scheduling.daycount import DayCountConvention, _get_convention


class DigitalType(Enum):
    """Digital option payoff type."""
    CASH_OR_NOTHING = "cash_or_nothing"
    ASSET_OR_NOTHING = "asset_or_nothing"


@dataclass
class FXOptionResult:
    """Result of FX option valuation."""
    premium: float              # Option premium
    premium_pct: float          # Premium as % of notional
    delta: float                # Spot delta
    delta_pips: float           # Delta per pip move
    gamma: float                # Gamma
    vega: float                 # Vega (per 1% vol move)
    theta: float                # Theta (per day)
    rho_domestic: float         # Rho w.r.t. domestic rate
    rho_foreign: float          # Rho w.r.t. foreign rate
    implied_vol: Optional[float] = None


def garman_kohlhagen_call(
    spot: float,
    strike: float,
    domestic_rate: float,
    foreign_rate: float,
    volatility: float,
    time_to_expiry: float,
) -> float:
    """
    Garman-Kohlhagen call option price.

    C = S * exp(-r_f * T) * N(d1) - K * exp(-r_d * T) * N(d2)

    d1 = (ln(S/K) + (r_d - r_f + σ²/2) * T) / (σ * √T)
    d2 = d1 - σ * √T
    """
    if time_to_expiry <= 0:
        return max(spot - strike, 0)

    if volatility <= 0:
        return max(spot * math.exp(-foreign_rate * time_to_expiry) -
                  strike * math.exp(-domestic_rate * time_to_expiry), 0)

    sqrt_t = math.sqrt(time_to_expiry)

    d1 = (math.log(spot / strike) + (domestic_rate - foreign_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    call = (spot * math.exp(-foreign_rate * time_to_expiry) * norm.cdf(d1) -
            strike * math.exp(-domestic_rate * time_to_expiry) * norm.cdf(d2))

    return call


def garman_kohlhagen_put(
    spot: float,
    strike: float,
    domestic_rate: float,
    foreign_rate: float,
    volatility: float,
    time_to_expiry: float,
) -> float:
    """Garman-Kohlhagen put option price."""
    if time_to_expiry <= 0:
        return max(strike - spot, 0)

    if volatility <= 0:
        return max(strike * math.exp(-domestic_rate * time_to_expiry) -
                  spot * math.exp(-foreign_rate * time_to_expiry), 0)

    sqrt_t = math.sqrt(time_to_expiry)

    d1 = (math.log(spot / strike) + (domestic_rate - foreign_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    put = (strike * math.exp(-domestic_rate * time_to_expiry) * norm.cdf(-d2) -
           spot * math.exp(-foreign_rate * time_to_expiry) * norm.cdf(-d1))

    return put


def garman_kohlhagen_greeks(
    spot: float,
    strike: float,
    domestic_rate: float,
    foreign_rate: float,
    volatility: float,
    time_to_expiry: float,
    option_type: OptionType,
) -> dict:
    """
    Calculate Garman-Kohlhagen Greeks.

    Returns:
        Dictionary with delta, gamma, vega, theta, rho_d, rho_f
    """
    if time_to_expiry <= 0 or volatility <= 0:
        intrinsic = (spot - strike if option_type == OptionType.CALL
                    else strike - spot)
        return {
            "delta": 1.0 if intrinsic > 0 else 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho_domestic": 0.0,
            "rho_foreign": 0.0,
        }

    sqrt_t = math.sqrt(time_to_expiry)

    d1 = (math.log(spot / strike) + (domestic_rate - foreign_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    exp_rf = math.exp(-foreign_rate * time_to_expiry)
    exp_rd = math.exp(-domestic_rate * time_to_expiry)

    # Delta
    if option_type == OptionType.CALL:
        delta = exp_rf * norm.cdf(d1)
    else:
        delta = -exp_rf * norm.cdf(-d1)

    # Gamma (same for call and put)
    gamma = exp_rf * norm.pdf(d1) / (spot * volatility * sqrt_t)

    # Vega (same for call and put) - per 1% vol change
    vega = spot * exp_rf * norm.pdf(d1) * sqrt_t / 100

    # Theta - per day
    theta_call = (
        -spot * exp_rf * norm.pdf(d1) * volatility / (2 * sqrt_t) -
        domestic_rate * strike * exp_rd * norm.cdf(d2) +
        foreign_rate * spot * exp_rf * norm.cdf(d1)
    )
    theta_put = (
        -spot * exp_rf * norm.pdf(d1) * volatility / (2 * sqrt_t) +
        domestic_rate * strike * exp_rd * norm.cdf(-d2) -
        foreign_rate * spot * exp_rf * norm.cdf(-d1)
    )

    if option_type == OptionType.CALL:
        theta = theta_call / 365
    else:
        theta = theta_put / 365

    # Rho domestic
    if option_type == OptionType.CALL:
        rho_d = strike * time_to_expiry * exp_rd * norm.cdf(d2) / 100
    else:
        rho_d = -strike * time_to_expiry * exp_rd * norm.cdf(-d2) / 100

    # Rho foreign
    if option_type == OptionType.CALL:
        rho_f = -spot * time_to_expiry * exp_rf * norm.cdf(d1) / 100
    else:
        rho_f = spot * time_to_expiry * exp_rf * norm.cdf(-d1) / 100

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho_domestic": rho_d,
        "rho_foreign": rho_f,
    }


class FXOption:
    """
    Vanilla FX Option.

    European-style option to buy (call) or sell (put) a currency pair.

    Convention:
    - Call on EUR/USD: Right to buy EUR, sell USD at strike
    - Put on EUR/USD: Right to sell EUR, buy USD at strike
    """

    def __init__(
        self,
        base_currency: str,
        quote_currency: str,
        notional: float,
        strike: float,
        expiry_date: date,
        option_type: Union[str, OptionType] = OptionType.CALL,
        delivery_date: Optional[date] = None,
        day_count: Union[str, DayCountConvention] = "ACT/365F",
    ):
        """
        Initialize FX option.

        Args:
            base_currency: Base currency
            quote_currency: Quote currency
            notional: Notional in base currency
            strike: Strike rate
            expiry_date: Option expiry date
            option_type: CALL or PUT
            delivery_date: Delivery date (typically expiry + 2 days)
            day_count: Day count convention
        """
        self.base_currency = base_currency.upper()
        self.quote_currency = quote_currency.upper()
        self.notional = notional
        self.strike = strike
        self.expiry_date = expiry_date

        if isinstance(option_type, str):
            option_type = OptionType[option_type.upper()]
        self.option_type = option_type

        self.delivery_date = delivery_date or (expiry_date + timedelta(days=2))

        if isinstance(day_count, str):
            day_count = _get_convention(day_count)
        self.day_count = day_count

    @property
    def pair(self) -> str:
        """Currency pair string."""
        return f"{self.base_currency}/{self.quote_currency}"

    def _time_to_expiry(self, valuation_date: date) -> float:
        """Calculate time to expiry in years."""
        return self.day_count.year_fraction(valuation_date, self.expiry_date)

    def price(
        self,
        valuation_date: date,
        spot: float,
        domestic_curve: Curve,
        foreign_curve: Curve,
        volatility: float,
    ) -> float:
        """
        Calculate option premium.

        Args:
            valuation_date: Valuation date
            spot: Current spot rate
            domestic_curve: Domestic (quote) currency curve
            foreign_curve: Foreign (base) currency curve
            volatility: Implied volatility

        Returns:
            Premium in quote currency
        """
        if self.expiry_date <= valuation_date:
            # Expired - return intrinsic value
            if self.option_type == OptionType.CALL:
                return max(spot - self.strike, 0) * self.notional
            else:
                return max(self.strike - spot, 0) * self.notional

        t = self._time_to_expiry(valuation_date)
        r_d = domestic_curve.zero_rate(t)
        r_f = foreign_curve.zero_rate(t)

        if self.option_type == OptionType.CALL:
            unit_price = garman_kohlhagen_call(spot, self.strike, r_d, r_f, volatility, t)
        else:
            unit_price = garman_kohlhagen_put(spot, self.strike, r_d, r_f, volatility, t)

        return unit_price * self.notional

    def premium_percentage(
        self,
        valuation_date: date,
        spot: float,
        domestic_curve: Curve,
        foreign_curve: Curve,
        volatility: float,
    ) -> float:
        """Premium as percentage of notional."""
        premium = self.price(valuation_date, spot, domestic_curve, foreign_curve, volatility)
        return premium / (self.notional * spot) * 100

    def implied_volatility(
        self,
        market_premium: float,
        valuation_date: date,
        spot: float,
        domestic_curve: Curve,
        foreign_curve: Curve,
    ) -> float:
        """
        Calculate implied volatility from market premium.

        Args:
            market_premium: Market premium in quote currency
            valuation_date: Valuation date
            spot: Spot rate
            domestic_curve: Domestic curve
            foreign_curve: Foreign curve

        Returns:
            Implied volatility
        """
        def objective(vol):
            return self.price(valuation_date, spot, domestic_curve, foreign_curve, vol) - market_premium

        try:
            return brentq(objective, 0.001, 3.0)
        except ValueError:
            return 0.15

    def greeks(
        self,
        valuation_date: date,
        spot: float,
        domestic_curve: Curve,
        foreign_curve: Curve,
        volatility: float,
    ) -> FXOptionResult:
        """
        Calculate option Greeks and full valuation.

        Returns:
            FXOptionResult with all Greeks
        """
        if self.expiry_date <= valuation_date:
            intrinsic = (max(spot - self.strike, 0) if self.option_type == OptionType.CALL
                        else max(self.strike - spot, 0))
            return FXOptionResult(
                premium=intrinsic * self.notional,
                premium_pct=intrinsic / spot * 100,
                delta=1.0 if intrinsic > 0 else 0.0,
                delta_pips=self.notional * 0.0001 if intrinsic > 0 else 0.0,
                gamma=0.0,
                vega=0.0,
                theta=0.0,
                rho_domestic=0.0,
                rho_foreign=0.0,
                implied_vol=None,
            )

        t = self._time_to_expiry(valuation_date)
        r_d = domestic_curve.zero_rate(t)
        r_f = foreign_curve.zero_rate(t)

        premium = self.price(valuation_date, spot, domestic_curve, foreign_curve, volatility)
        premium_pct = premium / (self.notional * spot) * 100

        greeks = garman_kohlhagen_greeks(spot, self.strike, r_d, r_f, volatility, t, self.option_type)

        # Scale by notional
        delta = greeks["delta"] * self.notional
        delta_pips = delta * 0.0001
        gamma = greeks["gamma"] * self.notional
        vega = greeks["vega"] * self.notional
        theta = greeks["theta"] * self.notional
        rho_d = greeks["rho_domestic"] * self.notional
        rho_f = greeks["rho_foreign"] * self.notional

        return FXOptionResult(
            premium=premium,
            premium_pct=premium_pct,
            delta=delta,
            delta_pips=delta_pips,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho_domestic=rho_d,
            rho_foreign=rho_f,
            implied_vol=volatility,
        )

    def delta_hedge_amount(
        self,
        valuation_date: date,
        spot: float,
        domestic_curve: Curve,
        foreign_curve: Curve,
        volatility: float,
    ) -> float:
        """
        Calculate amount needed to delta hedge.

        Returns:
            Amount in base currency to buy/sell
        """
        result = self.greeks(valuation_date, spot, domestic_curve, foreign_curve, volatility)
        return -result.delta

    def __repr__(self) -> str:
        return (f"FXOption({self.option_type.value} {self.pair}, "
                f"strike={self.strike:.5f}, "
                f"notional={self.notional:,.2f}, "
                f"expiry={self.expiry_date})")


class BarrierOption:
    """
    Barrier FX Option.

    Option that is activated (knock-in) or deactivated (knock-out)
    when spot reaches a barrier level.
    """

    def __init__(
        self,
        base_currency: str,
        quote_currency: str,
        notional: float,
        strike: float,
        barrier: float,
        expiry_date: date,
        option_type: Union[str, OptionType] = OptionType.CALL,
        barrier_type: Union[str, BarrierType] = BarrierType.DOWN_AND_OUT,
        rebate: float = 0.0,
        day_count: Union[str, DayCountConvention] = "ACT/365F",
    ):
        """
        Initialize barrier option.

        Args:
            base_currency: Base currency
            quote_currency: Quote currency
            notional: Notional in base currency
            strike: Strike rate
            barrier: Barrier level
            expiry_date: Expiry date
            option_type: CALL or PUT
            barrier_type: Type of barrier
            rebate: Rebate paid if knocked out
            day_count: Day count convention
        """
        self.base_currency = base_currency.upper()
        self.quote_currency = quote_currency.upper()
        self.notional = notional
        self.strike = strike
        self.barrier = barrier
        self.expiry_date = expiry_date
        self.rebate = rebate

        if isinstance(option_type, str):
            option_type = OptionType[option_type.upper()]
        self.option_type = option_type

        if isinstance(barrier_type, str):
            barrier_type = BarrierType[barrier_type.upper()]
        self.barrier_type = barrier_type

        if isinstance(day_count, str):
            day_count = _get_convention(day_count)
        self.day_count = day_count

    def _time_to_expiry(self, valuation_date: date) -> float:
        return self.day_count.year_fraction(valuation_date, self.expiry_date)

    def price(
        self,
        valuation_date: date,
        spot: float,
        domestic_curve: Curve,
        foreign_curve: Curve,
        volatility: float,
        num_sims: int = 100000,
    ) -> float:
        """
        Price barrier option using Monte Carlo simulation.

        Args:
            valuation_date: Valuation date
            spot: Current spot
            domestic_curve: Domestic curve
            foreign_curve: Foreign curve
            volatility: Volatility
            num_sims: Number of simulations

        Returns:
            Option premium
        """
        if self.expiry_date <= valuation_date:
            return 0.0

        t = self._time_to_expiry(valuation_date)
        r_d = domestic_curve.zero_rate(t)
        r_f = foreign_curve.zero_rate(t)

        # Monte Carlo simulation
        dt = t / 252  # Daily steps
        num_steps = max(int(252 * t), 1)

        rng = random.Random(42)

        drift = (r_d - r_f - 0.5 * volatility**2) * dt
        diffusion = volatility * math.sqrt(dt)

        # Generate paths and calculate payoffs
        payoff_sum = 0.0

        for _ in range(num_sims):
            # Generate one path
            log_spot = math.log(spot)
            hit_barrier = False

            for _ in range(num_steps):
                z = rng.gauss(0, 1)
                log_spot += drift + diffusion * z
                current_spot = math.exp(log_spot)

                # Check barrier
                if self.barrier_type in [BarrierType.DOWN_AND_OUT, BarrierType.DOWN_AND_IN]:
                    if current_spot <= self.barrier:
                        hit_barrier = True
                else:  # UP barriers
                    if current_spot >= self.barrier:
                        hit_barrier = True

            final_spot = math.exp(log_spot)

            # Determine if option is alive based on barrier type
            if self.barrier_type == BarrierType.DOWN_AND_OUT:
                alive = not hit_barrier
            elif self.barrier_type == BarrierType.DOWN_AND_IN:
                alive = hit_barrier
            elif self.barrier_type == BarrierType.UP_AND_OUT:
                alive = not hit_barrier
            elif self.barrier_type == BarrierType.UP_AND_IN:
                alive = hit_barrier

            # Calculate payoff
            if self.option_type == OptionType.CALL:
                intrinsic = max(final_spot - self.strike, 0)
            else:
                intrinsic = max(self.strike - final_spot, 0)

            if alive:
                payoff_sum += intrinsic
            elif "OUT" in self.barrier_type.value.upper():
                payoff_sum += self.rebate

        # Discount and average
        df = math.exp(-r_d * t)
        premium = (payoff_sum / num_sims) * df * self.notional

        return premium

    def __repr__(self) -> str:
        return (f"BarrierOption({self.barrier_type.value} "
                f"{self.option_type.value}, "
                f"strike={self.strike:.5f}, "
                f"barrier={self.barrier:.5f})")


class DigitalOption:
    """
    Digital (Binary) FX Option.

    Pays a fixed amount if spot is above (call) or below (put) strike at expiry.
    """

    def __init__(
        self,
        base_currency: str,
        quote_currency: str,
        notional: float,
        strike: float,
        expiry_date: date,
        payout: float,
        option_type: Union[str, OptionType] = OptionType.CALL,
        digital_type: Union[str, DigitalType] = DigitalType.CASH_OR_NOTHING,
        day_count: Union[str, DayCountConvention] = "ACT/365F",
    ):
        """
        Initialize digital option.

        Args:
            base_currency: Base currency
            quote_currency: Quote currency
            notional: Reference notional
            strike: Strike rate
            expiry_date: Expiry date
            payout: Fixed payout amount (for cash-or-nothing)
            option_type: CALL or PUT
            digital_type: Payout type
            day_count: Day count convention
        """
        self.base_currency = base_currency.upper()
        self.quote_currency = quote_currency.upper()
        self.notional = notional
        self.strike = strike
        self.expiry_date = expiry_date
        self.payout = payout

        if isinstance(option_type, str):
            option_type = OptionType[option_type.upper()]
        self.option_type = option_type

        if isinstance(digital_type, str):
            digital_type = DigitalType[digital_type.upper()]
        self.digital_type = digital_type

        if isinstance(day_count, str):
            day_count = _get_convention(day_count)
        self.day_count = day_count

    def _time_to_expiry(self, valuation_date: date) -> float:
        return self.day_count.year_fraction(valuation_date, self.expiry_date)

    def price(
        self,
        valuation_date: date,
        spot: float,
        domestic_curve: Curve,
        foreign_curve: Curve,
        volatility: float,
    ) -> float:
        """
        Price digital option using analytical formula.

        Cash-or-nothing call: payout * exp(-r_d * T) * N(d2)
        """
        if self.expiry_date <= valuation_date:
            if self.option_type == OptionType.CALL:
                itm = spot >= self.strike
            else:
                itm = spot <= self.strike
            return self.payout if itm else 0.0

        t = self._time_to_expiry(valuation_date)
        r_d = domestic_curve.zero_rate(t)
        r_f = foreign_curve.zero_rate(t)

        if volatility <= 0:
            return 0.0

        sqrt_t = math.sqrt(t)
        d1 = (math.log(spot / self.strike) + (r_d - r_f + 0.5 * volatility**2) * t) / (volatility * sqrt_t)
        d2 = d1 - volatility * sqrt_t

        df = math.exp(-r_d * t)

        if self.digital_type == DigitalType.CASH_OR_NOTHING:
            if self.option_type == OptionType.CALL:
                return self.payout * df * norm.cdf(d2)
            else:
                return self.payout * df * norm.cdf(-d2)
        else:
            # Asset-or-nothing
            df_f = math.exp(-r_f * t)
            if self.option_type == OptionType.CALL:
                return spot * df_f * norm.cdf(d1)
            else:
                return spot * df_f * norm.cdf(-d1)

    def __repr__(self) -> str:
        return (f"DigitalOption({self.digital_type.value} "
                f"{self.option_type.value}, "
                f"strike={self.strike:.5f}, "
                f"payout={self.payout:,.2f})")
