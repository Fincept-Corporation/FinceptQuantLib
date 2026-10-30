"""
Money Market Instruments

Provides:
- Cash deposits
- Certificates of Deposit (CDs)
- Commercial Paper
- Repurchase Agreements (Repos)
- Treasury Bills
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional
from enum import Enum
import math

from finceptquantlib.curves.curve import Curve
from finceptquantlib.scheduling.daycount import DayCountConvention, Actual360


class DepositType(Enum):
    """Type of deposit instrument."""
    OVERNIGHT = "overnight"
    TOM_NEXT = "tom_next"
    SPOT_NEXT = "spot_next"
    TERM = "term"


@dataclass
class DepositResult:
    """Result of deposit calculation."""
    present_value: float
    accrued_interest: float
    yield_to_maturity: float
    days_to_maturity: int


class Deposit:
    """
    Cash deposit / money market deposit.

    Simple interest instrument with single payment at maturity.
    """

    def __init__(
        self,
        notional: float,
        rate: float,
        start_date: date,
        maturity_date: date,
        day_count: Optional[DayCountConvention] = None,
        currency: str = "USD",
    ):
        self.notional = notional
        self.rate = rate
        self.start_date = start_date
        self.maturity_date = maturity_date
        self._day_count = day_count or Actual360()
        self.currency = currency

    @classmethod
    def overnight(
        cls,
        notional: float,
        rate: float,
        value_date: date,
        day_count: Optional[DayCountConvention] = None,
    ) -> "Deposit":
        """Create overnight deposit."""
        return cls(
            notional=notional,
            rate=rate,
            start_date=value_date,
            maturity_date=value_date + timedelta(days=1),
            day_count=day_count,
        )

    @classmethod
    def term(
        cls,
        notional: float,
        rate: float,
        value_date: date,
        tenor_days: int,
        day_count: Optional[DayCountConvention] = None,
    ) -> "Deposit":
        """Create term deposit."""
        return cls(
            notional=notional,
            rate=rate,
            start_date=value_date,
            maturity_date=value_date + timedelta(days=tenor_days),
            day_count=day_count,
        )

    def year_fraction(self) -> float:
        """Calculate year fraction for accrual period."""
        return self._day_count.year_fraction(self.start_date, self.maturity_date)

    def maturity_amount(self) -> float:
        """Calculate amount received at maturity."""
        yf = self.year_fraction()
        return self.notional * (1 + self.rate * yf)

    def present_value(
        self,
        valuation_date: date,
        discount_curve: Curve,
    ) -> float:
        """
        Calculate present value.

        Args:
            valuation_date: Date of valuation
            discount_curve: Discount curve

        Returns:
            Present value
        """
        if valuation_date >= self.maturity_date:
            return self.maturity_amount()

        t = (self.maturity_date - valuation_date).days / 365.0
        df = discount_curve.discount_factor(t)

        return self.maturity_amount() * df

    def accrued_interest(self, valuation_date: date) -> float:
        """Calculate accrued interest to valuation date."""
        if valuation_date <= self.start_date:
            return 0.0
        if valuation_date >= self.maturity_date:
            return self.notional * self.rate * self.year_fraction()

        accrual_yf = self._day_count.year_fraction(self.start_date, valuation_date)
        return self.notional * self.rate * accrual_yf

    def implied_rate(
        self,
        valuation_date: date,
        discount_curve: Curve,
    ) -> float:
        """Calculate implied deposit rate from curve."""
        t_start = max(0, (self.start_date - valuation_date).days / 365.0)
        t_end = (self.maturity_date - valuation_date).days / 365.0

        df_start = discount_curve.discount_factor(t_start)
        df_end = discount_curve.discount_factor(t_end)

        yf = self.year_fraction()
        return (df_start / df_end - 1) / yf


class CertificateOfDeposit:
    """
    Certificate of Deposit (CD).

    Negotiable time deposit with fixed maturity.
    """

    def __init__(
        self,
        face_value: float,
        coupon_rate: float,
        issue_date: date,
        maturity_date: date,
        day_count: Optional[DayCountConvention] = None,
    ):
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.issue_date = issue_date
        self.maturity_date = maturity_date
        self._day_count = day_count or Actual360()

    def maturity_value(self) -> float:
        """Value at maturity including interest."""
        yf = self._day_count.year_fraction(self.issue_date, self.maturity_date)
        return self.face_value * (1 + self.coupon_rate * yf)

    def price(
        self,
        valuation_date: date,
        market_yield: float,
    ) -> float:
        """
        Calculate CD price from market yield.

        Args:
            valuation_date: Date of valuation
            market_yield: Market yield for discounting

        Returns:
            Clean price
        """
        maturity_val = self.maturity_value()
        days_to_maturity = (self.maturity_date - valuation_date).days
        yf = days_to_maturity / 360.0

        return maturity_val / (1 + market_yield * yf)

    def yield_to_maturity(
        self,
        valuation_date: date,
        price: float,
    ) -> float:
        """
        Calculate yield from price.

        Args:
            valuation_date: Date of valuation
            price: Market price

        Returns:
            Yield to maturity
        """
        maturity_val = self.maturity_value()
        days_to_maturity = (self.maturity_date - valuation_date).days
        yf = days_to_maturity / 360.0

        return (maturity_val / price - 1) / yf


class CommercialPaper:
    """
    Commercial Paper (CP).

    Short-term unsecured promissory note.
    Typically issued at discount.
    """

    def __init__(
        self,
        face_value: float,
        issue_date: date,
        maturity_date: date,
        discount_rate: Optional[float] = None,
        price: Optional[float] = None,
    ):
        self.face_value = face_value
        self.issue_date = issue_date
        self.maturity_date = maturity_date

        if discount_rate is not None:
            self._discount_rate = discount_rate
            self._issue_price = self._calculate_price_from_rate()
        elif price is not None:
            self._issue_price = price
            self._discount_rate = self._calculate_rate_from_price()
        else:
            raise ValueError("Must provide either discount_rate or price")

    def _calculate_price_from_rate(self) -> float:
        """Calculate price from discount rate."""
        days = (self.maturity_date - self.issue_date).days
        return self.face_value * (1 - self._discount_rate * days / 360)

    def _calculate_rate_from_price(self) -> float:
        """Calculate discount rate from price."""
        days = (self.maturity_date - self.issue_date).days
        return (self.face_value - self._issue_price) / self.face_value * 360 / days

    @property
    def discount_rate(self) -> float:
        """Discount rate (bank discount yield)."""
        return self._discount_rate

    @property
    def issue_price(self) -> float:
        """Issue price."""
        return self._issue_price

    def bond_equivalent_yield(self) -> float:
        """
        Calculate bond-equivalent yield.

        BEY = (Face - Price) / Price * 365 / Days
        """
        days = (self.maturity_date - self.issue_date).days
        return (self.face_value - self._issue_price) / self._issue_price * 365 / days

    def money_market_yield(self) -> float:
        """
        Calculate money market yield (CD equivalent).

        MMY = (Face - Price) / Price * 360 / Days
        """
        days = (self.maturity_date - self.issue_date).days
        return (self.face_value - self._issue_price) / self._issue_price * 360 / days


class Repo:
    """
    Repurchase Agreement (Repo).

    Agreement to sell securities and repurchase at a higher price.
    """

    def __init__(
        self,
        principal: float,
        repo_rate: float,
        start_date: date,
        end_date: date,
        collateral_value: float,
        haircut: float = 0.02,
        day_count: Optional[DayCountConvention] = None,
    ):
        self.principal = principal
        self.repo_rate = repo_rate
        self.start_date = start_date
        self.end_date = end_date
        self.collateral_value = collateral_value
        self.haircut = haircut
        self._day_count = day_count or Actual360()

    @property
    def margin(self) -> float:
        """Initial margin (collateral / principal)."""
        return self.collateral_value / self.principal

    def repo_interest(self) -> float:
        """Calculate repo interest."""
        yf = self._day_count.year_fraction(self.start_date, self.end_date)
        return self.principal * self.repo_rate * yf

    def repurchase_price(self) -> float:
        """Price at which securities are repurchased."""
        return self.principal + self.repo_interest()

    def implied_rate(
        self,
        start_price: float,
        end_price: float,
    ) -> float:
        """
        Calculate implied repo rate from prices.

        Args:
            start_price: Sale price
            end_price: Repurchase price

        Returns:
            Implied repo rate
        """
        yf = self._day_count.year_fraction(self.start_date, self.end_date)
        return (end_price / start_price - 1) / yf

    def margin_call(self, current_collateral_value: float) -> float:
        """
        Calculate margin call amount.

        Args:
            current_collateral_value: Current value of collateral

        Returns:
            Additional margin required (negative if excess)
        """
        required_margin = self.principal * (1 + self.haircut)
        return required_margin - current_collateral_value


class TreasuryBill:
    """
    Treasury Bill (T-Bill).

    Short-term government security issued at discount.
    """

    def __init__(
        self,
        face_value: float,
        issue_date: date,
        maturity_date: date,
        discount_rate: Optional[float] = None,
        price: Optional[float] = None,
    ):
        self.face_value = face_value
        self.issue_date = issue_date
        self.maturity_date = maturity_date

        if discount_rate is not None:
            self._discount_rate = discount_rate
        elif price is not None:
            days = (self.maturity_date - self.issue_date).days
            self._discount_rate = (self.face_value - price) / self.face_value * 360 / days
        else:
            raise ValueError("Must provide either discount_rate or price")

    @property
    def discount_rate(self) -> float:
        """Bank discount rate."""
        return self._discount_rate

    def price(self, settlement_date: Optional[date] = None) -> float:
        """
        Calculate T-Bill price.

        Args:
            settlement_date: Settlement date (default: issue date)

        Returns:
            Price per $100 face
        """
        if settlement_date is None:
            settlement_date = self.issue_date

        days = (self.maturity_date - settlement_date).days
        return self.face_value * (1 - self._discount_rate * days / 360)

    def bond_equivalent_yield(self, settlement_date: Optional[date] = None) -> float:
        """
        Calculate bond-equivalent yield.

        For T-Bills <= 182 days:
        BEY = (Face - Price) / Price * 365 / Days

        For T-Bills > 182 days, use more complex formula.
        """
        if settlement_date is None:
            settlement_date = self.issue_date

        days = (self.maturity_date - settlement_date).days
        p = self.price(settlement_date)

        if days <= 182:
            return (self.face_value - p) / p * 365 / days
        else:
            # More complex formula for longer bills
            # BEY = (-2n/m + 2*sqrt((n/m)^2 - (2n/m - 1)(1 - 100/P))) / (2n/m - 1)
            # where n = days, m = 365
            n_m = days / 365
            ratio = self.face_value / p

            discriminant = n_m ** 2 - (2 * n_m - 1) * (1 - ratio)
            if discriminant < 0:
                return (self.face_value - p) / p * 365 / days

            return (-2 * n_m + 2 * math.sqrt(discriminant)) / (2 * n_m - 1)

    def money_market_yield(self, settlement_date: Optional[date] = None) -> float:
        """Calculate money market yield (360-day basis)."""
        if settlement_date is None:
            settlement_date = self.issue_date

        days = (self.maturity_date - settlement_date).days
        p = self.price(settlement_date)

        return (self.face_value - p) / p * 360 / days

    def discount_to_yield(self, discount_rate: float, days: int) -> float:
        """Convert discount rate to bond-equivalent yield."""
        price_per_100 = 100 * (1 - discount_rate * days / 360)
        return (100 - price_per_100) / price_per_100 * 365 / days

    def yield_to_discount(self, bey: float, days: int) -> float:
        """Convert bond-equivalent yield to discount rate."""
        return bey / (1 + bey * days / 365) * 360 / days
