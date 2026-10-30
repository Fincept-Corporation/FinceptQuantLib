"""
Commodity Derivatives

Derivatives on physical commodities:
- Commodity swaps
- Commodity futures
- Commodity options
- Spread options
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional, List
import math


class CommodityType(Enum):
    """Commodity types."""
    ENERGY = "energy"  # Oil, natural gas
    METALS = "metals"  # Gold, silver, copper
    AGRICULTURE = "agriculture"  # Corn, wheat, soybeans
    LIVESTOCK = "livestock"  # Cattle, hogs


@dataclass
class CommodityIndex:
    """Commodity price index."""
    name: str
    commodity_type: CommodityType
    current_price: float
    unit: str = "BBL"  # Barrels, Troy oz, Bushels, etc.


class CommoditySwap:
    """
    Commodity Swap.

    Exchange fixed price for floating commodity price.

    Examples:
        >>> swap = CommoditySwap(
        ...     commodity="WTI Crude",
        ...     notional_quantity=10000,  # Barrels
        ...     fixed_price=75.0,  # $/barrel
        ...     start_date=date(2024, 1, 1),
        ...     end_date=date(2025, 1, 1),
        ...     payment_frequency=12  # Monthly
        ... )
        >>> npv = swap.present_value(
        ...     spot_price=80.0,
        ...     forward_prices=[80.0] * 12,
        ...     discount_factors=[0.996 ** (i + 1) for i in range(12)],
        ... )
        >>> round(npv, 2)
        48718.88
    """

    def __init__(
        self,
        commodity: str,
        notional_quantity: float,
        fixed_price: float,
        start_date: date,
        end_date: date,
        payment_frequency: int = 12,  # Monthly
        pay_fixed: bool = True
    ) -> None:
        """
        Initialize commodity swap.

        Args:
            commodity: Commodity name
            notional_quantity: Physical quantity
            fixed_price: Fixed price per unit
            start_date: Start date
            end_date: End date
            payment_frequency: Payments per year
            pay_fixed: If True, pay fixed receive floating
        """
        self.commodity = commodity
        self.notional_quantity = notional_quantity
        self.fixed_price = fixed_price
        self.start_date = start_date
        self.end_date = end_date
        self.payment_frequency = payment_frequency
        self.pay_fixed = pay_fixed

    def present_value(
        self,
        spot_price: float,
        forward_prices: List[float],
        discount_factors: List[float],
        convenience_yield: float = 0.0
    ) -> float:
        """
        Calculate swap NPV.

        Args:
            spot_price: Current spot price
            forward_prices: Forward prices for each payment date
            discount_factors: Discount factors
            convenience_yield: Convenience yield %

        Returns:
            Net present value
        """
        n_payments = len(forward_prices)
        pv_fixed = 0.0
        pv_float = 0.0

        quantity_per_period = self.notional_quantity / n_payments

        for i in range(n_payments):
            fwd_price = forward_prices[i] if i < len(forward_prices) else spot_price
            df = discount_factors[i] if i < len(discount_factors) else 1.0

            pv_fixed += self.fixed_price * quantity_per_period * df
            pv_float += fwd_price * quantity_per_period * df

        if self.pay_fixed:
            return pv_float - pv_fixed
        else:
            return pv_fixed - pv_float


class CommodityOption:
    """
    Commodity Option.

    European or American option on commodity price.

    Examples:
        >>> option = CommodityOption(
        ...     commodity="Gold",
        ...     strike=2000,
        ...     expiry=date(2024, 12, 31),
        ...     quantity=100,  # Troy ounces
        ...     is_call=True
        ... )
        >>> price = option.price(spot=2100, volatility=0.20, risk_free_rate=0.04)
    """

    def __init__(
        self,
        commodity: str,
        strike: float,
        expiry: date,
        quantity: float,
        is_call: bool = True,
        is_american: bool = False
    ) -> None:
        """Initialize commodity option."""
        self.commodity = commodity
        self.strike = strike
        self.expiry = expiry
        self.quantity = quantity
        self.is_call = is_call
        self.is_american = is_american

    def price(
        self,
        spot: float,
        volatility: float,
        risk_free_rate: float,
        convenience_yield: float = 0.0,
        time_to_expiry: Optional[float] = None
    ) -> float:
        """
        Price using Black-76 model (commodity futures option).

        Args:
            spot: Spot commodity price
            volatility: Price volatility
            risk_free_rate: Risk-free rate
            convenience_yield: Convenience yield
            time_to_expiry: Time to expiry in years

        Returns:
            Option price
        """
        if time_to_expiry is None:
            time_to_expiry = 1.0  # Simplified

        from scipy.stats import norm

        # Forward price
        F = spot * math.exp((risk_free_rate - convenience_yield) * time_to_expiry)
        K = self.strike
        T = time_to_expiry
        sigma = volatility

        # Black-76 formula
        d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        disc = math.exp(-risk_free_rate * T)

        if self.is_call:
            option_value = disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            option_value = disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

        return option_value * self.quantity


class CommodityFuture:
    """Commodity futures contract."""

    def __init__(
        self,
        commodity: str,
        contract_size: float,
        delivery_date: date,
        futures_price: float
    ) -> None:
        """Initialize commodity future."""
        self.commodity = commodity
        self.contract_size = contract_size
        self.delivery_date = delivery_date
        self.futures_price = futures_price

    def mark_to_market(self, current_futures_price: float, position_size: int = 1) -> float:
        """
        Calculate mark-to-market value.

        Args:
            current_futures_price: Current futures price
            position_size: Number of contracts (+ for long, - for short)

        Returns:
            MTM P&L
        """
        price_change = current_futures_price - self.futures_price
        return price_change * self.contract_size * position_size


class SpreadOption:
    """
    Commodity spread option (e.g., crack spread, spark spread).

    Option on the spread between two commodities.
    """

    def __init__(
        self,
        commodity1: str,
        commodity2: str,
        strike_spread: float,
        expiry: date,
        quantity: float,
        is_call: bool = True
    ) -> None:
        """
        Initialize spread option.

        Args:
            commodity1: First commodity
            commodity2: Second commodity
            strike_spread: Strike on spread (price1 - price2)
            expiry: Expiry date
            quantity: Contract size
            is_call: Call on spread
        """
        self.commodity1 = commodity1
        self.commodity2 = commodity2
        self.strike_spread = strike_spread
        self.expiry = expiry
        self.quantity = quantity
        self.is_call = is_call

    def price(
        self,
        spot1: float,
        spot2: float,
        vol1: float,
        vol2: float,
        correlation: float,
        risk_free_rate: float,
        time_to_expiry: float
    ) -> float:
        """
        Price spread option using Kirk's approximation.

        Args:
            spot1: Spot price commodity 1
            spot2: Spot price commodity 2
            vol1: Volatility commodity 1
            vol2: Volatility commodity 2
            correlation: Correlation between commodities
            risk_free_rate: Risk-free rate
            time_to_expiry: Time to expiry (years)

        Returns:
            Option price
        """
        from scipy.stats import norm

        # Kirk's approximation for spread option
        # Reference: Kirk (1995) - Correlation in the Energy Markets
        F1 = spot1
        F2 = spot2
        K = self.strike_spread
        T = time_to_expiry

        # Kirk's uses F2 + K as denominator to avoid div-by-zero
        F2_K = F2 + K * math.exp(-risk_free_rate * T)
        if F2_K <= 0:
            # Degenerate case: intrinsic value
            disc = math.exp(-risk_free_rate * T)
            spread = F1 - F2
            if self.is_call:
                return disc * max(spread - K, 0) * self.quantity
            else:
                return disc * max(K - spread, 0) * self.quantity

        ratio = F2 / F2_K
        sigma = math.sqrt(vol1**2 + (ratio * vol2)**2 - 2 * correlation * vol1 * vol2 * ratio)

        if sigma * math.sqrt(T) < 1e-10:
            disc = math.exp(-risk_free_rate * T)
            if self.is_call:
                return disc * max(F1 - F2_K, 0) * self.quantity
            else:
                return disc * max(F2_K - F1, 0) * self.quantity

        d1 = (math.log(F1 / F2_K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        disc = math.exp(-risk_free_rate * T)

        if self.is_call:
            value = disc * (F1 * norm.cdf(d1) - F2_K * norm.cdf(d2))
        else:
            value = disc * (F2_K * norm.cdf(-d2) - F1 * norm.cdf(-d1))

        return value * self.quantity
