"""
Securitized Products

Mortgage-Backed Securities (MBS), Asset-Backed Securities (ABS),
and Structured Products (CLO, CDO).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import List, Dict


class MBSType(Enum):
    """MBS product types."""
    PASS_THROUGH = "pass_through"
    CMO = "cmo"  # Collateralized Mortgage Obligation
    TBA = "tba"  # To Be Announced


class PrepaymentModel(Enum):
    """Prepayment models."""
    PSA = "psa"  # Public Securities Association
    CPR = "cpr"  # Constant Prepayment Rate
    SMM = "smm"  # Single Monthly Mortality


@dataclass
class MortgagePool:
    """
    Pool of mortgages underlying MBS.

    Attributes:
        original_balance: Original pool balance
        current_balance: Current outstanding balance
        weighted_avg_coupon: WAC (%)
        weighted_avg_maturity: WAM (months)
        weighted_avg_loan_age: WALA (months)
        loan_to_value: Average LTV ratio
        num_loans: Number of loans in pool
    """
    original_balance: float
    current_balance: float
    weighted_avg_coupon: float
    weighted_avg_maturity: float
    weighted_avg_loan_age: float = 0.0
    loan_to_value: float = 0.80
    num_loans: int = 1000

    @property
    def pool_factor(self) -> float:
        """Pool factor (current / original balance)."""
        return self.current_balance / self.original_balance if self.original_balance > 0 else 0


class MortgageBackedSecurity:
    """
    Mortgage-Backed Security (MBS).

    Pass-through security backed by pool of mortgages.

    Examples:
        >>> pool = MortgagePool(
        ...     original_balance=100_000_000,
        ...     current_balance=95_000_000,
        ...     weighted_avg_coupon=4.5,
        ...     weighted_avg_maturity=300
        ... )
        >>> mbs = MortgageBackedSecurity(
        ...     pool=pool,
        ...     pass_through_rate=4.0,  # Lower than WAC (servicing fee)
        ...     issue_date=date(2024, 1, 1)
        ... )
        >>> cashflows = mbs.generate_cashflows(prepayment_rate=0.06)
    """

    def __init__(
        self,
        pool: MortgagePool,
        pass_through_rate: float,
        issue_date: date,
        mbs_type: MBSType = MBSType.PASS_THROUGH
    ) -> None:
        """
        Initialize MBS.

        Args:
            pool: Underlying mortgage pool
            pass_through_rate: Rate passed through to investors (< WAC)
            issue_date: Issue date
            mbs_type: Type of MBS
        """
        self.pool = pool
        self.pass_through_rate = pass_through_rate
        self.issue_date = issue_date
        self.mbs_type = mbs_type

    def generate_cashflows(
        self,
        prepayment_rate: float,
        months: int = 360
    ) -> List[Dict[str, float]]:
        """
        Generate monthly cash flows with prepayments.

        Args:
            prepayment_rate: CPR (Constant Prepayment Rate, annualized %)
            months: Number of months to project

        Returns:
            List of monthly cash flows with principal and interest
        """
        cashflows = []
        balance = self.pool.current_balance
        monthly_rate = self.pass_through_rate / 100 / 12

        # SMM from CPR
        smm = 1 - (1 - prepayment_rate) ** (1/12)

        for month in range(1, months + 1):
            if balance < 1:
                break

            # Scheduled principal
            scheduled_principal = self._scheduled_principal(balance, monthly_rate, months - month + 1)

            # Interest payment
            interest = balance * monthly_rate

            # Prepayment
            prepayment = (balance - scheduled_principal) * smm

            # Total principal
            total_principal = scheduled_principal + prepayment

            cashflows.append({
                "month": month,
                "beginning_balance": balance,
                "scheduled_principal": scheduled_principal,
                "prepayment": prepayment,
                "total_principal": total_principal,
                "interest": interest,
                "total_cashflow": total_principal + interest,
                "ending_balance": balance - total_principal
            })

            balance -= total_principal

        return cashflows

    def _scheduled_principal(self, balance: float, rate: float, n: int) -> float:
        """Calculate scheduled monthly principal payment."""
        if rate == 0:
            return balance / n

        payment = balance * rate / (1 - (1 + rate) ** -n)
        interest = balance * rate
        return payment - interest

    def price(
        self,
        discount_rate: float,
        prepayment_rate: float
    ) -> float:
        """
        Calculate MBS price (clean price).

        Args:
            discount_rate: Discount rate for PV
            prepayment_rate: CPR assumption

        Returns:
            MBS price per $100 par
        """
        cashflows = self.generate_cashflows(prepayment_rate)
        monthly_disc = discount_rate / 12

        pv = 0.0
        for cf in cashflows:
            disc_factor = 1 / ((1 + monthly_disc) ** cf["month"])
            pv += cf["total_cashflow"] * disc_factor

        # Price per $100
        return (pv / self.pool.current_balance) * 100


class AssetBackedSecurity:
    """
    Asset-Backed Security (ABS).

    Securities backed by pools of assets (auto loans, credit cards, etc.).
    """

    def __init__(
        self,
        pool_balance: float,
        coupon_rate: float,
        weighted_avg_life: float,
        credit_enhancement: float = 0.05
    ) -> None:
        """
        Initialize ABS.

        Args:
            pool_balance: Pool balance
            coupon_rate: Coupon rate
            weighted_avg_life: WAL in years
            credit_enhancement: Credit enhancement % (subordination, overcollateralization)
        """
        self.pool_balance = pool_balance
        self.coupon_rate = coupon_rate
        self.weighted_avg_life = weighted_avg_life
        self.credit_enhancement = credit_enhancement

    def price(self, discount_spread: float) -> float:
        """
        Price ABS.

        Simplified pricing using WAL and spread.
        """
        # Present value of cash flows
        discount_rate = discount_spread
        pv = self.pool_balance / ((1 + discount_rate) ** self.weighted_avg_life)

        return (pv / self.pool_balance) * 100


@dataclass
class Tranche:
    """
    CDO/CLO tranche.

    Attributes:
        name: Tranche name (Senior, Mezzanine, Equity)
        notional: Tranche size
        attachment_point: Loss % where tranche starts taking losses
        detachment_point: Loss % where tranche fully wiped out
        coupon_rate: Tranche coupon
        seniority: Seniority level (1 = most senior)
    """
    name: str
    notional: float
    attachment_point: float  # %
    detachment_point: float  # %
    coupon_rate: float
    seniority: int

    def loss_allocation(self, pool_loss_pct: float) -> float:
        """
        Calculate tranche loss given pool loss percentage.

        Args:
            pool_loss_pct: Pool loss as % of original balance

        Returns:
            Tranche loss amount
        """
        if pool_loss_pct <= self.attachment_point:
            return 0.0
        elif pool_loss_pct >= self.detachment_point:
            return self.notional
        else:
            # Proportional loss within tranche
            tranche_width = self.detachment_point - self.attachment_point
            loss_into_tranche = pool_loss_pct - self.attachment_point
            loss_pct = loss_into_tranche / tranche_width if tranche_width > 0 else 0
            return self.notional * loss_pct


class CDO:
    """
    Collateralized Debt Obligation (CDO).

    Structured product with multiple tranches of varying seniority.

    Examples:
        >>> tranches = [
        ...     Tranche("Senior AAA", 70_000_000, 0.08, 1.00, 0.03, 1),
        ...     Tranche("Mezzanine BBB", 20_000_000, 0.03, 0.08, 0.06, 2),
        ...     Tranche("Equity", 10_000_000, 0.00, 0.03, 0.15, 3)
        ... ]
        >>> cdo = CDO(tranches=tranches, reference_pool_size=100_000_000)
        >>> value = cdo.tranche_value("Senior AAA", expected_loss_pct=0.02)
    """

    def __init__(
        self,
        tranches: List[Tranche],
        reference_pool_size: float
    ) -> None:
        """
        Initialize CDO.

        Args:
            tranches: List of tranches (ordered by seniority)
            reference_pool_size: Size of reference pool
        """
        self.tranches = sorted(tranches, key=lambda t: t.seniority)
        self.reference_pool_size = reference_pool_size

    def tranche_value(
        self,
        tranche_name: str,
        expected_loss_pct: float,
        discount_rate: float = 0.05,
        time_horizon: float = 5.0
    ) -> float:
        """
        Value a specific tranche.

        Args:
            tranche_name: Name of tranche to value
            expected_loss_pct: Expected pool loss %
            discount_rate: Discount rate
            time_horizon: Time horizon in years

        Returns:
            Tranche value
        """
        tranche = next((t for t in self.tranches if t.name == tranche_name), None)
        if not tranche:
            raise ValueError(f"Tranche {tranche_name} not found")

        # Expected loss on tranche
        tranche_loss = tranche.loss_allocation(expected_loss_pct)

        # Expected cash flows
        expected_principal = tranche.notional - tranche_loss
        expected_coupons = tranche.notional * tranche.coupon_rate * time_horizon

        # PV
        pv_principal = expected_principal / ((1 + discount_rate) ** time_horizon)
        pv_coupons = expected_coupons / ((1 + discount_rate) ** (time_horizon / 2))  # Mid-point

        return pv_principal + pv_coupons


class CLO(CDO):
    """
    Collateralized Loan Obligation.

    CDO backed by leveraged loans.
    """

    def __init__(
        self,
        tranches: List[Tranche],
        reference_pool_size: float,
        manager_fee: float = 0.005
    ) -> None:
        """
        Initialize CLO.

        Args:
            tranches: Tranches
            reference_pool_size: Loan pool size
            manager_fee: Annual management fee %
        """
        super().__init__(tranches, reference_pool_size)
        self.manager_fee = manager_fee
