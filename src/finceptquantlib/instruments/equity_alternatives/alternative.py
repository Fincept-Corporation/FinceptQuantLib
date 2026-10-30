"""
Alternative Risk Transfer (ART) Instruments

Insurance-linked securities and weather derivatives:
- Weather derivatives
- Catastrophe bonds (cat bonds)
- Insurance swaps
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional, List
import math


class WeatherIndex(Enum):
    """Weather measurement indices."""
    HDD = "heating_degree_days"  # Heating Degree Days
    CDD = "cooling_degree_days"  # Cooling Degree Days
    RAINFALL = "rainfall"  # Cumulative rainfall
    SNOWFALL = "snowfall"
    WIND_SPEED = "wind_speed"
    TEMPERATURE = "temperature"


@dataclass
class WeatherMeasurement:
    """Weather observation."""
    date: date
    value: float
    index_type: WeatherIndex


class WeatherDerivative:
    """
    Weather Derivative.

    Derivative with payoff linked to weather index (HDD, CDD, rainfall, etc.).

    Examples:
        >>> weather = WeatherDerivative(
        ...     index_type=WeatherIndex.HDD,
        ...     strike=1000,  # 1000 heating degree days
        ...     tick_value=10000,  # $10k per HDD point
        ...     start_date=date(2024, 11, 1),
        ...     end_date=date(2025, 3, 31),
        ...     location="Chicago"
        ... )
        >>> payoff = weather.payoff(realized_index=1200)  # Cold winter
    """

    def __init__(
        self,
        index_type: WeatherIndex,
        strike: float,
        tick_value: float,
        start_date: date,
        end_date: date,
        location: str,
        is_call: bool = True,
        is_swap: bool = False
    ) -> None:
        """
        Initialize weather derivative.

        Args:
            index_type: Type of weather index
            strike: Strike level for index
            tick_value: Dollar value per index point
            start_date: Measurement period start
            end_date: Measurement period end
            location: Geographic location
            is_call: True for call (pays if index > strike), False for put
            is_swap: True for swap, False for option
        """
        self.index_type = index_type
        self.strike = strike
        self.tick_value = tick_value
        self.start_date = start_date
        self.end_date = end_date
        self.location = location
        self.is_call = is_call
        self.is_swap = is_swap

    def payoff(self, realized_index: float) -> float:
        """
        Calculate payoff based on realized index.

        Args:
            realized_index: Cumulative index value over measurement period

        Returns:
            Payoff amount
        """
        diff = realized_index - self.strike

        if self.is_swap:
            # Swap: pay difference (can be negative)
            return self.tick_value * diff
        else:
            # Option: max(diff, 0) for call, max(-diff, 0) for put
            if self.is_call:
                return self.tick_value * max(diff, 0)
            else:
                return self.tick_value * max(-diff, 0)

    def price(
        self,
        expected_index: float,
        index_volatility: float,
        discount_rate: float = 0.03,
        time_to_expiry: Optional[float] = None
    ) -> float:
        """
        Price weather derivative.

        For options: uses Black-76 on weather index.
        For swaps: discounted expected payoff.

        Args:
            expected_index: Expected cumulative index
            index_volatility: Index volatility
            discount_rate: Discount rate
            time_to_expiry: Time in years

        Returns:
            Derivative price
        """
        if time_to_expiry is None:
            time_to_expiry = (self.end_date - self.start_date).days / 365.25

        if self.is_swap:
            # Swap value
            expected_payoff = self.tick_value * (expected_index - self.strike)
            return expected_payoff / ((1 + discount_rate) ** time_to_expiry)
        else:
            # Option value using Black-76
            from scipy.stats import norm

            F = expected_index
            K = self.strike
            T = time_to_expiry
            sigma = index_volatility

            if sigma * math.sqrt(T) < 1e-6:
                # No volatility: intrinsic value
                return max((F - K) * self.tick_value, 0) if self.is_call else max((K - F) * self.tick_value, 0)

            d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)

            disc = math.exp(-discount_rate * T)

            if self.is_call:
                value = disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
            else:
                value = disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

            return value * self.tick_value


class CatastropheBond:
    """
    Catastrophe Bond (Cat Bond).

    Insurance-linked security with principal at risk from catastrophic events.

    Examples:
        >>> cat_bond = CatastropheBond(
        ...     issue_date=date(2024, 1, 1),
        ...     maturity_date=date(2027, 1, 1),
        ...     par_value=100_000_000,
        ...     coupon_rate=0.08,  # High coupon for cat risk
        ...     trigger_type="indemnity",
        ...     attachment_point=500_000_000,  # Losses must exceed this
        ...     exhaustion_point=1_000_000_000,
        ...     covered_perils=["Hurricane", "Earthquake"]
        ... )
    """

    def __init__(
        self,
        issue_date: date,
        maturity_date: date,
        par_value: float,
        coupon_rate: float,
        trigger_type: str,  # "indemnity", "index", "parametric"
        attachment_point: float,
        exhaustion_point: float,
        covered_perils: List[str],
        geographic_region: str = "US"
    ) -> None:
        """
        Initialize catastrophe bond.

        Args:
            issue_date: Issue date
            maturity_date: Maturity date
            par_value: Par value
            coupon_rate: Coupon rate (typically high, 5-10%)
            trigger_type: Type of trigger mechanism
            attachment_point: Loss level where principal starts being lost
            exhaustion_point: Loss level where all principal lost
            covered_perils: List of covered catastrophes
            geographic_region: Geographic coverage
        """
        self.issue_date = issue_date
        self.maturity_date = maturity_date
        self.par_value = par_value
        self.coupon_rate = coupon_rate
        self.trigger_type = trigger_type
        self.attachment_point = attachment_point
        self.exhaustion_point = exhaustion_point
        self.covered_perils = covered_perils
        self.geographic_region = geographic_region

    def principal_loss(self, event_loss: float) -> float:
        """
        Calculate principal loss from catastrophe event.

        Args:
            event_loss: Insured loss from event

        Returns:
            Principal loss amount
        """
        if event_loss <= self.attachment_point:
            return 0.0
        elif event_loss >= self.exhaustion_point:
            return self.par_value
        else:
            # Pro-rata loss
            loss_into_bond = event_loss - self.attachment_point
            total_coverage = self.exhaustion_point - self.attachment_point
            loss_pct = loss_into_bond / total_coverage if total_coverage > 0 else 0
            return self.par_value * loss_pct

    def expected_loss(self, annual_cat_probability: float, average_severity: float) -> float:
        """
        Calculate expected principal loss.

        Args:
            annual_cat_probability: Probability of triggering event per year
            average_severity: Average event loss given occurrence

        Returns:
            Expected loss amount
        """
        years = (self.maturity_date - self.issue_date).days / 365.25

        # Probability of event over life of bond
        prob_event = 1 - (1 - annual_cat_probability) ** years

        # Expected loss given event
        expected_loss_given_event = self.principal_loss(average_severity)

        return prob_event * expected_loss_given_event

    def price(
        self,
        risk_free_rate: float,
        credit_spread: float,
        cat_risk_premium: float,
        annual_cat_prob: float,
        expected_severity: float,
        valuation_date: Optional[date] = None
    ) -> float:
        """
        Price catastrophe bond.

        Args:
            risk_free_rate: Risk-free rate
            credit_spread: Issuer credit spread
            cat_risk_premium: Premium for catastrophe risk
            annual_cat_prob: Annual probability of trigger
            expected_severity: Expected loss severity
            valuation_date: Valuation date

        Returns:
            Cat bond price
        """
        if valuation_date is None:
            valuation_date = date.today()

        years = (self.maturity_date - valuation_date).days / 365.25

        # Expected principal recovery
        expected_loss = self.expected_loss(annual_cat_prob, expected_severity)
        expected_principal = self.par_value - expected_loss

        # Discount rate includes cat risk premium
        discount_rate = risk_free_rate + credit_spread + cat_risk_premium

        # PV of expected cash flows
        # Coupons (assuming paid if no event)
        pv_coupons = (self.par_value * self.coupon_rate *
                     (1 - (1 + discount_rate) ** -years) / discount_rate)

        # Principal
        pv_principal = expected_principal / ((1 + discount_rate) ** years)

        total_pv = pv_coupons + pv_principal

        return (total_pv / self.par_value) * 100  # Price per 100


def degree_days(temperatures: List[float], base_temp: float = 65.0, heating: bool = True) -> float:
    """
    Calculate degree days from temperature observations.

    Args:
        temperatures: Daily average temperatures (Fahrenheit)
        base_temp: Base temperature (typically 65°F)
        heating: True for HDD, False for CDD

    Returns:
        Cumulative degree days
    """
    total = 0.0
    for temp in temperatures:
        if heating:
            # HDD: max(base - temp, 0)
            total += max(base_temp - temp, 0)
        else:
            # CDD: max(temp - base, 0)
            total += max(temp - base_temp, 0)

    return total
