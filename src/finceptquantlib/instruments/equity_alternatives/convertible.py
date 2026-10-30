"""
Convertible Bonds

Hybrid securities combining bond and equity features:
- Convertible bonds (CB)
- Convertible bond pricing (binomial tree, Monte Carlo)
- Conversion features (ratio, price, forced conversion)
- Call/put provisions
- Greeks (delta, gamma, vega, rho)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional, List
import math


class ConversionTrigger(Enum):
    """Conversion trigger type."""
    VOLUNTARY = "voluntary"
    MANDATORY = "mandatory"
    CONTINGENT = "contingent"


@dataclass
class ConversionTerms:
    """
    Conversion terms for convertible bond.

    Attributes:
        conversion_ratio: Shares per bond
        conversion_price: Effective stock price for conversion
        conversion_premium: Premium over current stock price (%)
        conversion_start: First date conversion allowed
        conversion_end: Last date conversion allowed
        trigger: Conversion trigger type
    """
    conversion_ratio: float
    conversion_price: float
    conversion_premium: float = 0.0
    conversion_start: Optional[date] = None
    conversion_end: Optional[date] = None
    trigger: ConversionTrigger = ConversionTrigger.VOLUNTARY

    @classmethod
    def from_premium(
        cls,
        par_value: float,
        current_stock_price: float,
        conversion_premium_pct: float = 25.0
    ) -> ConversionTerms:
        """
        Create conversion terms from premium.

        Args:
            par_value: Bond par value
            current_stock_price: Current stock price
            conversion_premium_pct: Premium % (e.g., 25 = 25%)

        Returns:
            ConversionTerms
        """
        conversion_price = current_stock_price * (1 + conversion_premium_pct / 100)
        conversion_ratio = par_value / conversion_price

        return cls(
            conversion_ratio=conversion_ratio,
            conversion_price=conversion_price,
            conversion_premium=conversion_premium_pct
        )


@dataclass
class CallProvision:
    """Call provision allowing issuer to redeem early."""
    call_price: float
    call_start_date: date
    soft_call_trigger: Optional[float] = None  # Stock price for soft call


@dataclass
class PutProvision:
    """Put provision allowing holder to sell back."""
    put_price: float
    put_dates: List[date]


class ConvertibleBond:
    """
    Convertible Bond.

    Hybrid security with bond and equity characteristics.
    Can be converted to equity at holder's option (or issuer mandate).

    Examples:
        >>> from finceptquantlib.scheduling import Frequency
        >>> conv_terms = ConversionTerms.from_premium(1000, 50, conversion_premium_pct=25)
        >>> cb = ConvertibleBond(
        ...     issue_date=date(2024, 1, 1),
        ...     maturity_date=date(2029, 1, 1),
        ...     par_value=1000,
        ...     coupon_rate=0.02,  # Low coupon due to conversion feature
        ...     conversion_terms=conv_terms,
        ...     frequency=Frequency.SEMI_ANNUAL
        ... )
        >>> price = cb.price(stock_price=60, credit_spread=0.03, volatility=0.30)
    """

    def __init__(
        self,
        issue_date: date,
        maturity_date: date,
        par_value: float,
        coupon_rate: float,
        conversion_terms: ConversionTerms,
        frequency,  # Frequency enum
        call_provision: Optional[CallProvision] = None,
        put_provision: Optional[PutProvision] = None,
        day_count: str = "30/360",
        currency: str = "USD"
    ) -> None:
        """
        Initialize convertible bond.

        Args:
            issue_date: Issue date
            maturity_date: Maturity date
            par_value: Par/face value
            coupon_rate: Annual coupon rate (typically low, 0-3%)
            conversion_terms: Conversion parameters
            frequency: Coupon frequency
            call_provision: Optional call provision
            put_provision: Optional put provision
            day_count: Day count convention
            currency: Currency
        """
        self.issue_date = issue_date
        self.maturity_date = maturity_date
        self.par_value = par_value
        self.coupon_rate = coupon_rate
        self.conversion_terms = conversion_terms
        self.frequency = frequency
        self.call_provision = call_provision
        self.put_provision = put_provision
        self.day_count = day_count
        self.currency = currency

    def price(
        self,
        stock_price: float,
        credit_spread: float,
        volatility: float,
        risk_free_rate: float = 0.03,
        dividend_yield: float = 0.0,
        valuation_date: Optional[date] = None
    ) -> dict:
        """
        Price convertible bond using binomial tree.

        Args:
            stock_price: Current stock price
            credit_spread: Credit spread over risk-free
            volatility: Stock volatility
            risk_free_rate: Risk-free rate
            dividend_yield: Dividend yield
            valuation_date: Valuation date

        Returns:
            dict with price, bond_floor, parity, conversion_value
        """
        if valuation_date is None:
            valuation_date = date.today()

        # Bond floor (straight bond value)
        bond_floor = self._bond_floor(credit_spread, risk_free_rate, valuation_date)

        # Conversion parity (conversion value)
        conversion_value = stock_price * self.conversion_terms.conversion_ratio

        # Binomial tree pricing (simplified)
        cb_price = self._binomial_tree_price(
            stock_price, volatility, risk_free_rate, dividend_yield,
            credit_spread, valuation_date
        )

        return {
            "price": cb_price,
            "bond_floor": bond_floor,
            "conversion_value": conversion_value,
            "parity": conversion_value,
            "premium_over_bond": ((cb_price / bond_floor) - 1) * 100 if bond_floor > 0 else 0,
            "premium_over_parity": ((cb_price / conversion_value) - 1) * 100 if conversion_value > 0 else 0
        }

    def _bond_floor(self, credit_spread: float, risk_free_rate: float, valuation_date: date) -> float:
        """Calculate straight bond value (bond floor)."""
        # Simplified: PV of coupons + par
        years_to_maturity = (self.maturity_date - valuation_date).days / 365.25
        ytm = risk_free_rate + credit_spread

        periods_per_year = self.frequency.value if hasattr(self.frequency, 'value') else 2
        n_periods = int(years_to_maturity * periods_per_year)

        coupon_payment = self.par_value * self.coupon_rate / periods_per_year
        discount_rate = ytm / periods_per_year

        # PV of coupons
        if discount_rate == 0:
            pv_coupons = coupon_payment * n_periods
        else:
            pv_coupons = coupon_payment * (1 - (1 + discount_rate) ** -n_periods) / discount_rate

        # PV of par
        pv_par = self.par_value / ((1 + discount_rate) ** n_periods)

        return pv_coupons + pv_par

    def _binomial_tree_price(
        self,
        stock_price: float,
        volatility: float,
        risk_free_rate: float,
        dividend_yield: float,
        credit_spread: float,
        valuation_date: date,
        n_steps: int = 100
    ) -> float:
        """
        Price using binomial tree.

        Combines bond value and conversion option.
        """
        years_to_maturity = (self.maturity_date - valuation_date).days / 365.25
        dt = years_to_maturity / n_steps

        # Stock price tree parameters
        u = math.exp(volatility * math.sqrt(dt))
        d = 1 / u
        p = (math.exp((risk_free_rate - dividend_yield) * dt) - d) / (u - d)

        # Discount factor
        disc = math.exp(-(risk_free_rate + credit_spread) * dt)

        # Initialize tree at maturity
        conversion_ratio = self.conversion_terms.conversion_ratio
        stock_prices = [stock_price * (u ** (n_steps - 2*i)) for i in range(n_steps + 1)]

        values = []
        for S in stock_prices:
            # At maturity: max(par, conversion value)
            conversion_val = S * conversion_ratio
            bond_val = self.par_value
            values.append(max(bond_val, conversion_val))

        # Work backwards
        for step in range(n_steps - 1, -1, -1):
            new_values = []

            for i in range(step + 1):
                # Expected value
                continuation = (p * values[i] + (1 - p) * values[i + 1]) * disc

                # Conversion value at this node
                S = stock_price * (u ** (step - 2*i))
                conversion_val = S * conversion_ratio

                # Value = max(continuation, conversion)
                node_value = max(continuation, conversion_val)

                # Check call provision
                if self.call_provision and step > 0:
                    node_value = min(node_value, self.call_provision.call_price)

                new_values.append(node_value)

            values = new_values

        return values[0]

    def conversion_probability(
        self,
        stock_price: float,
        volatility: float,
        years_to_maturity: float
    ) -> float:
        """
        Estimate probability of conversion.

        Uses Black-Scholes framework.
        """
        from scipy.stats import norm

        # Probability stock exceeds conversion price
        conversion_price = self.conversion_terms.conversion_price
        moneyness = math.log(stock_price / conversion_price)

        drift = -0.5 * volatility**2 * years_to_maturity
        std = volatility * math.sqrt(years_to_maturity)

        z = -(moneyness + drift) / std if std > 0 else 0
        prob = norm.cdf(-z)

        return prob
