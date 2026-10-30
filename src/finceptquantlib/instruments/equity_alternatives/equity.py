"""
Equity Derivatives

Equity-linked derivatives:
- Equity swaps
- Total return swaps
- Variance swaps
- Volatility swaps
- Equity forwards
"""

from __future__ import annotations

from datetime import date
from typing import List, Optional
import math


class EquitySwap:
    """
    Equity Swap.

    Exchange equity returns for fixed/floating rate.

    Examples:
        >>> eq_swap = EquitySwap(
        ...     notional=10_000_000,
        ...     equity_index="SPX",
        ...     start_date=date(2024, 1, 1),
        ...     end_date=date(2025, 1, 1),
        ...     fixed_rate=0.03,
        ...     pay_equity_return=False
        ... )
        >>> npv = eq_swap.present_value(equity_return=0.15, discount_rate=0.04)
    """

    def __init__(
        self,
        notional: float,
        equity_index: str,
        start_date: date,
        end_date: date,
        fixed_rate: Optional[float] = None,
        floating_spread: float = 0.0,
        pay_equity_return: bool = True,
        frequency: int = 4  # Quarterly
    ) -> None:
        """
        Initialize equity swap.

        Args:
            notional: Notional amount
            equity_index: Equity index name
            start_date: Start date
            end_date: End date
            fixed_rate: Fixed rate (if None, floating)
            floating_spread: Spread over floating rate
            pay_equity_return: If True, pay equity receive fixed/float
            frequency: Payment frequency per year
        """
        self.notional = notional
        self.equity_index = equity_index
        self.start_date = start_date
        self.end_date = end_date
        self.fixed_rate = fixed_rate
        self.floating_spread = floating_spread
        self.pay_equity_return = pay_equity_return
        self.frequency = frequency

    def present_value(
        self,
        equity_return: float,
        discount_rate: float,
        floating_rate: Optional[float] = None
    ) -> float:
        """
        Calculate NPV.

        Args:
            equity_return: Total equity return over period (%)
            discount_rate: Discount rate
            floating_rate: Floating rate (if applicable)

        Returns:
            NPV
        """
        # Equity leg PV
        pv_equity = self.notional * equity_return

        # Fixed/Floating leg PV
        if self.fixed_rate is not None:
            rate = self.fixed_rate
        elif floating_rate is not None:
            rate = floating_rate + self.floating_spread
        else:
            rate = 0.04  # Default

        years = (self.end_date - self.start_date).days / 365.25
        pv_fixed = self.notional * rate * years

        # Discount to present
        pv_equity = pv_equity / ((1 + discount_rate) ** years)
        pv_fixed = pv_fixed / ((1 + discount_rate) ** years)

        if self.pay_equity_return:
            return pv_fixed - pv_equity
        else:
            return pv_equity - pv_fixed


class TotalReturnSwap:
    """
    Total Return Swap (TRS).

    Exchange total return (price change + dividends/coupons) of asset
    for fixed/floating rate.

    Examples:
        >>> trs = TotalReturnSwap(
        ...     reference_asset="AAPL",
        ...     notional=5_000_000,
        ...     start_date=date(2024, 1, 1),
        ...     end_date=date(2025, 1, 1),
        ...     financing_spread=0.0050  # 50 bps
        ... )
    """

    def __init__(
        self,
        reference_asset: str,
        notional: float,
        start_date: date,
        end_date: date,
        financing_spread: float = 0.0,
        pay_total_return: bool = False
    ) -> None:
        """
        Initialize TRS.

        Args:
            reference_asset: Reference asset (stock, bond, index)
            notional: Notional principal
            start_date: Start date
            end_date: End date
            financing_spread: Spread over reference rate
            pay_total_return: Direction
        """
        self.reference_asset = reference_asset
        self.notional = notional
        self.start_date = start_date
        self.end_date = end_date
        self.financing_spread = financing_spread
        self.pay_total_return = pay_total_return

    def present_value(
        self,
        price_return: float,
        income_return: float,
        reference_rate: float,
        discount_rate: float
    ) -> float:
        """
        Calculate TRS NPV.

        Args:
            price_return: Capital appreciation %
            income_return: Dividend/coupon yield %
            reference_rate: Reference funding rate (e.g., SOFR)
            discount_rate: Discount rate

        Returns:
            NPV
        """
        total_return = price_return + income_return
        financing_cost = reference_rate + self.financing_spread

        years = (self.end_date - self.start_date).days / 365.25

        pv_total_return = self.notional * total_return / ((1 + discount_rate) ** years)
        pv_financing = self.notional * financing_cost * years / ((1 + discount_rate) ** years)

        if self.pay_total_return:
            return pv_financing - pv_total_return
        else:
            return pv_total_return - pv_financing


class VarianceSwap:
    """
    Variance Swap.

    Swap on realized variance vs strike variance.
    Pure volatility exposure (no delta).

    Examples:
        >>> var_swap = VarianceSwap(
        ...     notional_variance=1_000_000,  # Vega notional
        ...     strike_variance=0.04,  # 20% vol squared
        ...     start_date=date(2024, 1, 1),
        ...     end_date=date(2024, 12, 31)
        ... )
        >>> payoff = var_swap.payoff(realized_variance=0.0625)  # 25% realized vol
    """

    def __init__(
        self,
        notional_variance: float,
        strike_variance: float,
        start_date: date,
        end_date: date,
        underlying: str = "SPX"
    ) -> None:
        """
        Initialize variance swap.

        Args:
            notional_variance: Vega notional ($ per variance point)
            strike_variance: Strike variance (σ²)
            start_date: Start date
            end_date: End date
            underlying: Underlying index
        """
        self.notional_variance = notional_variance
        self.strike_variance = strike_variance
        self.start_date = start_date
        self.end_date = end_date
        self.underlying = underlying

    def payoff(self, realized_variance: float) -> float:
        """
        Calculate variance swap payoff.

        Payoff = Notional * (Realized Variance - Strike Variance)

        Args:
            realized_variance: Realized variance over period

        Returns:
            Payoff (can be negative)
        """
        return self.notional_variance * (realized_variance - self.strike_variance)

    def fair_strike(
        self,
        current_vol: float,
        vol_of_vol: float,
        risk_free_rate: float,
        time_to_expiry: float
    ) -> float:
        """
        Calculate fair strike variance.

        Uses replication portfolio of vanilla options.
        """
        # Simplified: use current implied variance with convexity adjustment
        implied_variance = current_vol ** 2

        # Convexity adjustment
        convexity_adj = (vol_of_vol ** 2) * implied_variance * time_to_expiry / 2

        fair_strike = implied_variance + convexity_adj

        return fair_strike


class VolatilitySwap:
    """
    Volatility Swap.

    Swap on realized volatility (σ) vs strike volatility.

    Similar to variance swap but linear in vol instead of quadratic.

    Examples:
        >>> vol_swap = VolatilitySwap(
        ...     notional_volatility=500_000,
        ...     strike_volatility=0.20,  # 20% vol
        ...     start_date=date(2024, 1, 1),
        ...     end_date=date(2024, 12, 31)
        ... )
        >>> payoff = vol_swap.payoff(realized_volatility=0.25)  # 25% realized
    """

    def __init__(
        self,
        notional_volatility: float,
        strike_volatility: float,
        start_date: date,
        end_date: date,
        underlying: str = "SPX"
    ) -> None:
        """Initialize volatility swap."""
        self.notional_volatility = notional_volatility
        self.strike_volatility = strike_volatility
        self.start_date = start_date
        self.end_date = end_date
        self.underlying = underlying

    def payoff(self, realized_volatility: float) -> float:
        """
        Calculate volatility swap payoff.

        Payoff = Notional * (Realized Vol - Strike Vol)

        Args:
            realized_volatility: Realized volatility (annualized)

        Returns:
            Payoff
        """
        return self.notional_volatility * (realized_volatility - self.strike_volatility)

    def variance_to_volatility_adjustment(self, variance_strike: float, vol_of_vol: float, time: float) -> float:
        """
        Convert variance swap strike to volatility swap strike.

        Vol Swap Strike ≈ sqrt(Var Swap Strike) - (vol_of_vol² / 8) * sqrt(Var Strike)
        """
        vol_from_var = math.sqrt(variance_strike)
        adjustment = (vol_of_vol ** 2 / 8) * vol_from_var * time
        return vol_from_var - adjustment


def calculate_realized_variance(returns: List[float], annualization_factor: float = 252) -> float:
    """
    Calculate realized variance from return series.

    Args:
        returns: List of returns
        annualization_factor: Trading days per year

    Returns:
        Annualized realized variance
    """
    if not returns:
        return 0.0

    # Sample variance
    n = len(returns)
    mean_return = sum(returns) / n
    variance = sum((r - mean_return) ** 2 for r in returns) / (n - 1) if n > 1 else 0

    # Annualize
    return variance * annualization_factor


def calculate_realized_volatility(returns: List[float], annualization_factor: float = 252) -> float:
    """Calculate realized volatility from returns."""
    variance = calculate_realized_variance(returns, annualization_factor)
    return math.sqrt(variance)
