"""
Interest Rate Futures

Implements:
- Short-Term Interest Rate (STIR) Futures (e.g., Eurodollar, SOFR futures)
- Bond Futures (e.g., Treasury futures, Bund futures)

Key Features:
- Convexity adjustment for STIR futures
- Bond futures basis and cheapest-to-deliver analysis
- Implied repo rate calculation
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Union, Optional, List
from enum import Enum

from finceptquantlib.curves.curve import Curve
from finceptquantlib.scheduling.daycount import DayCountConvention, _get_convention


class FuturesContractType(Enum):
    """Type of futures contract."""
    EURODOLLAR = "eurodollar"
    SOFR_3M = "sofr_3m"
    SOFR_1M = "sofr_1m"
    EURIBOR = "euribor"
    TREASURY = "treasury"
    BUND = "bund"
    GILT = "gilt"


@dataclass
class FuturesResult:
    """Result of futures valuation."""
    price: float
    implied_rate: float
    convexity_adjustment: float
    dv01: float
    basis: Optional[float] = None
    implied_repo: Optional[float] = None


@dataclass
class CTDResult:
    """Cheapest-to-deliver analysis result."""
    ctd_bond: str
    conversion_factor: float
    basis: float
    implied_repo: float
    gross_basis: float
    net_basis: float


class STIRFuture:
    """
    Short-Term Interest Rate (STIR) Future.

    Examples: Eurodollar futures, SOFR futures, Euribor futures.

    Price convention: 100 - Rate
    So a 5% rate corresponds to price of 95.00

    Key feature: Convexity adjustment between futures rate and forward rate
    due to daily margining.
    """

    def __init__(
        self,
        contract_date: date,
        contract_type: Union[str, FuturesContractType] = FuturesContractType.SOFR_3M,
        tick_size: float = 0.0025,  # Quarter basis point
        tick_value: float = 6.25,   # USD per tick per contract
        contract_size: float = 1_000_000,
        day_count: Union[str, DayCountConvention] = "ACT/360",
        tenor_days: int = 90,
    ):
        """
        Initialize STIR future.

        Args:
            contract_date: Contract expiry/settlement date
            contract_type: Type of STIR future
            tick_size: Minimum price move (in rate terms, e.g., 0.0025 = 0.25bp)
            tick_value: Dollar value per tick per contract
            contract_size: Notional per contract
            day_count: Day count convention
            tenor_days: Underlying rate tenor in days
        """
        self.contract_date = contract_date

        if isinstance(contract_type, str):
            contract_type = FuturesContractType[contract_type.upper()]
        self.contract_type = contract_type

        self.tick_size = tick_size
        self.tick_value = tick_value
        self.contract_size = contract_size

        if isinstance(day_count, str):
            day_count = _get_convention(day_count)
        self.day_count = day_count

        self.tenor_days = tenor_days
        self._rate_tenor = tenor_days / 360.0

    @property
    def rate_start_date(self) -> date:
        """Start date of underlying rate period."""
        return self.contract_date

    @property
    def rate_end_date(self) -> date:
        """End date of underlying rate period."""
        return self.contract_date + timedelta(days=self.tenor_days)

    def price_from_rate(self, rate: float) -> float:
        """Convert rate to futures price."""
        return 100.0 - rate * 100.0

    def rate_from_price(self, price: float) -> float:
        """Convert futures price to rate."""
        return (100.0 - price) / 100.0

    def forward_rate(self, curve: Curve) -> float:
        """Calculate forward rate from curve."""
        t1 = curve.time_to_maturity(self.rate_start_date)
        t2 = curve.time_to_maturity(self.rate_end_date)

        if t1 <= 0:
            t1 = 0.001
        if t2 <= t1:
            t2 = t1 + self._rate_tenor

        return curve.forward_rate(t1, t2)

    def convexity_adjustment(
        self,
        valuation_date: date,
        curve: Curve,
        volatility: float = 0.01,
    ) -> float:
        """
        Calculate convexity adjustment.

        The adjustment accounts for the difference between futures rate
        and forward rate due to daily margining.

        Approximate formula:
        CA = 0.5 * sigma^2 * t1 * t2

        where sigma is rate volatility, t1 is time to contract,
        t2 is time to rate end.

        Args:
            valuation_date: Valuation date
            curve: Yield curve
            volatility: Rate volatility (annualized)

        Returns:
            Convexity adjustment (futures rate - forward rate)
        """
        t1 = curve.time_to_maturity(self.rate_start_date)
        t2 = curve.time_to_maturity(self.rate_end_date)

        if t1 <= 0:
            return 0.0

        # Simple approximation
        ca = 0.5 * volatility ** 2 * t1 * t2

        return ca

    def implied_rate(
        self,
        futures_price: float,
        valuation_date: date,
        curve: Curve,
        volatility: float = 0.01,
        include_convexity: bool = True,
    ) -> float:
        """
        Calculate implied forward rate from futures price.

        Args:
            futures_price: Observed futures price
            valuation_date: Valuation date
            curve: Yield curve
            volatility: Rate volatility
            include_convexity: Whether to adjust for convexity

        Returns:
            Implied forward rate
        """
        futures_rate = self.rate_from_price(futures_price)

        if include_convexity:
            ca = self.convexity_adjustment(valuation_date, curve, volatility)
            return futures_rate - ca
        else:
            return futures_rate

    def theoretical_price(
        self,
        valuation_date: date,
        curve: Curve,
        volatility: float = 0.01,
        include_convexity: bool = True,
    ) -> float:
        """
        Calculate theoretical futures price.

        Args:
            valuation_date: Valuation date
            curve: Yield curve
            volatility: Rate volatility
            include_convexity: Whether to include convexity adjustment

        Returns:
            Theoretical futures price
        """
        fwd_rate = self.forward_rate(curve)

        if include_convexity:
            ca = self.convexity_adjustment(valuation_date, curve, volatility)
            futures_rate = fwd_rate + ca
        else:
            futures_rate = fwd_rate

        return self.price_from_rate(futures_rate)

    def dv01(self, num_contracts: int = 1) -> float:
        """
        Calculate DV01 per contract.

        For STIR futures, DV01 = Contract size * rate tenor * 0.0001
        """
        return num_contracts * self.contract_size * self._rate_tenor * 0.0001

    def profit_loss(
        self,
        entry_price: float,
        exit_price: float,
        num_contracts: int = 1,
        position: str = "long",
    ) -> float:
        """
        Calculate P&L for a futures position.

        Args:
            entry_price: Entry price
            exit_price: Exit price
            num_contracts: Number of contracts
            position: "long" or "short"

        Returns:
            Profit/loss in currency units
        """
        price_change = exit_price - entry_price  # In price points

        # Convert to ticks
        ticks = price_change / (self.tick_size * 100)  # tick_size is in rate terms

        pnl = ticks * self.tick_value * num_contracts

        if position.lower() == "short":
            pnl = -pnl

        return pnl

    def value(
        self,
        valuation_date: date,
        curve: Curve,
        volatility: float = 0.01,
    ) -> FuturesResult:
        """Full valuation."""
        fwd = self.forward_rate(curve)
        ca = self.convexity_adjustment(valuation_date, curve, volatility)
        price = self.theoretical_price(valuation_date, curve, volatility)
        implied = fwd + ca

        return FuturesResult(
            price=price,
            implied_rate=implied,
            convexity_adjustment=ca,
            dv01=self.dv01(),
        )

    def __repr__(self) -> str:
        return (f"STIRFuture({self.contract_type.value}, "
                f"{self.contract_date}, "
                f"size={self.contract_size:,.0f})")


class BondFuture:
    """
    Bond Future.

    Examples: US Treasury futures, Bund futures, Gilt futures.

    Key concepts:
    - Deliverable basket of bonds
    - Conversion factors
    - Cheapest-to-deliver (CTD)
    - Implied repo rate
    - Basis trading
    """

    def __init__(
        self,
        delivery_date: date,
        first_delivery_date: Optional[date] = None,
        last_delivery_date: Optional[date] = None,
        contract_type: Union[str, FuturesContractType] = FuturesContractType.TREASURY,
        contract_size: float = 100_000,
        tick_size: float = 1/32,  # 1/32 of a point
        tick_value: float = 31.25,
        coupon_rate: float = 0.06,  # Notional coupon for CF calculation
    ):
        """
        Initialize bond future.

        Args:
            delivery_date: Contract delivery month (first of month)
            first_delivery_date: First day of delivery month
            last_delivery_date: Last day of delivery month
            contract_type: Type of bond future
            contract_size: Contract size
            tick_size: Minimum price move
            tick_value: Dollar value per tick
            coupon_rate: Notional coupon for conversion factor
        """
        self.delivery_date = delivery_date
        self.first_delivery_date = first_delivery_date or delivery_date
        self.last_delivery_date = (
            last_delivery_date or
            (delivery_date + timedelta(days=30)).replace(day=1) - timedelta(days=1)
        )

        if isinstance(contract_type, str):
            contract_type = FuturesContractType[contract_type.upper()]
        self.contract_type = contract_type

        self.contract_size = contract_size
        self.tick_size = tick_size
        self.tick_value = tick_value
        self.coupon_rate = coupon_rate

        # Deliverable basket
        self._deliverables: List[dict] = []

    def add_deliverable(
        self,
        bond_id: str,
        clean_price: float,
        coupon: float,
        maturity: date,
        conversion_factor: Optional[float] = None,
    ) -> None:
        """
        Add a bond to the deliverable basket.

        Args:
            bond_id: Bond identifier
            clean_price: Current clean price
            coupon: Annual coupon rate
            maturity: Maturity date
            conversion_factor: Pre-calculated CF (will calculate if None)
        """
        if conversion_factor is None:
            conversion_factor = self.calculate_conversion_factor(
                coupon, maturity, self.delivery_date
            )

        self._deliverables.append({
            "bond_id": bond_id,
            "clean_price": clean_price,
            "coupon": coupon,
            "maturity": maturity,
            "conversion_factor": conversion_factor,
        })

    def calculate_conversion_factor(
        self,
        coupon: float,
        maturity: date,
        delivery_date: date,
    ) -> float:
        """
        Calculate conversion factor for a bond.

        The CF converts bond price to equivalent price based on
        notional coupon rate.

        Simplified formula for US Treasuries:
        CF = PV of bond cash flows at notional yield / 100

        Args:
            coupon: Bond coupon rate
            maturity: Bond maturity date
            delivery_date: Futures delivery date

        Returns:
            Conversion factor
        """
        # Time to maturity in years
        ttm = (maturity - delivery_date).days / 365.25

        # Simplified CF calculation (actual method varies by exchange)
        # Using approximation: CF = c/y * (1 - 1/(1+y)^n) + 1/(1+y)^n
        # where y = notional yield, c = coupon, n = years to maturity

        y = self.coupon_rate
        c = coupon
        n = ttm

        if n <= 0:
            return 1.0

        # Semi-annual periods
        n_periods = int(n * 2)
        y_semi = y / 2
        c_semi = c / 2

        cf = 0.0
        for i in range(1, n_periods + 1):
            cf += c_semi / ((1 + y_semi) ** i)
        cf += 1.0 / ((1 + y_semi) ** n_periods)

        return cf

    def gross_basis(
        self,
        bond_price: float,
        futures_price: float,
        conversion_factor: float,
    ) -> float:
        """
        Calculate gross basis.

        Gross Basis = Bond Price - Futures Price * Conversion Factor
        """
        return bond_price - futures_price * conversion_factor

    def net_basis(
        self,
        bond_price: float,
        futures_price: float,
        conversion_factor: float,
        accrued_interest: float,
        carry: float,
    ) -> float:
        """
        Calculate net basis.

        Net Basis = Gross Basis - Carry
        where Carry = Accrued Interest + Coupon Income - Financing Cost
        """
        gross = self.gross_basis(bond_price, futures_price, conversion_factor)
        return gross - carry

    def implied_repo_rate(
        self,
        bond_price: float,
        futures_price: float,
        conversion_factor: float,
        accrued_at_delivery: float,
        accrued_now: float,
        days_to_delivery: int,
        coupon_income: float = 0.0,
    ) -> float:
        """
        Calculate implied repo rate.

        The implied repo rate is the financing rate implied by the
        relationship between cash bond and futures price.

        IRR = (Invoice Price / Full Bond Price - 1) * (360/days)

        Args:
            bond_price: Clean bond price
            futures_price: Futures price
            conversion_factor: Bond's conversion factor
            accrued_at_delivery: Accrued interest at delivery
            accrued_now: Current accrued interest
            days_to_delivery: Days until delivery
            coupon_income: Any coupon payments received

        Returns:
            Implied repo rate (annualized)
        """
        if days_to_delivery <= 0:
            return 0.0

        # Invoice price at delivery
        invoice_price = futures_price * conversion_factor + accrued_at_delivery

        # Current full price
        full_price = bond_price + accrued_now

        # Total return including coupon
        total_return = (invoice_price + coupon_income - full_price) / full_price

        # Annualize
        irr = total_return * 360 / days_to_delivery

        return irr

    def cheapest_to_deliver(
        self,
        futures_price: float,
        repo_rate: float,
        valuation_date: date,
    ) -> Optional[CTDResult]:
        """
        Find cheapest-to-deliver bond.

        The CTD is the bond with highest implied repo rate (or lowest net basis).

        Args:
            futures_price: Current futures price
            repo_rate: Market repo rate for financing
            valuation_date: Current date

        Returns:
            CTDResult with CTD analysis
        """
        if not self._deliverables:
            return None

        best_irr = float("-inf")
        ctd = None

        for bond in self._deliverables:
            # Calculate IRR for each bond
            days = (self.first_delivery_date - valuation_date).days
            if days <= 0:
                continue

            # Simplified accrued calculation
            accrued_now = bond["coupon"] / 2 * (1 - days / 182.5) if days < 182.5 else 0
            accrued_delivery = bond["coupon"] / 2 * 0.5  # Approximate

            irr = self.implied_repo_rate(
                bond_price=bond["clean_price"],
                futures_price=futures_price,
                conversion_factor=bond["conversion_factor"],
                accrued_at_delivery=accrued_delivery,
                accrued_now=accrued_now,
                days_to_delivery=days,
            )

            if irr > best_irr:
                best_irr = irr
                gross = self.gross_basis(
                    bond["clean_price"], futures_price, bond["conversion_factor"]
                )
                # Net basis = gross basis - carry
                carry = (bond["clean_price"] + accrued_now) * repo_rate * days / 360
                net = gross - carry

                ctd = CTDResult(
                    ctd_bond=bond["bond_id"],
                    conversion_factor=bond["conversion_factor"],
                    basis=gross,
                    implied_repo=irr,
                    gross_basis=gross,
                    net_basis=net,
                )

        return ctd

    def theoretical_price(
        self,
        ctd_price: float,
        ctd_conversion_factor: float,
        ctd_accrued: float,
        repo_rate: float,
        days_to_delivery: int,
    ) -> float:
        """
        Calculate theoretical futures price from CTD.

        F = (Bond Full Price * (1 + r * t) - Accrued at Delivery) / CF

        Args:
            ctd_price: CTD clean price
            ctd_conversion_factor: CTD conversion factor
            ctd_accrued: CTD current accrued
            repo_rate: Repo rate
            days_to_delivery: Days to delivery

        Returns:
            Theoretical futures price
        """
        full_price = ctd_price + ctd_accrued

        # Forward price of bond
        fwd_full = full_price * (1 + repo_rate * days_to_delivery / 360)

        # Accrued at delivery (simplified)
        accrued_delivery = ctd_accrued + ctd_accrued * days_to_delivery / 182.5

        # Futures price
        return (fwd_full - accrued_delivery) / ctd_conversion_factor

    def dv01(self, num_contracts: int = 1) -> float:
        """
        Calculate approximate DV01.

        For bond futures, DV01 depends on CTD duration.
        Approximate: contract_size * duration / 10000
        """
        # Assume duration around 7 for 10-year futures
        assumed_duration = 7.0
        return num_contracts * self.contract_size * assumed_duration / 10000

    def __repr__(self) -> str:
        return (f"BondFuture({self.contract_type.value}, "
                f"{self.delivery_date}, "
                f"deliverables={len(self._deliverables)})")


class FuturesStrip:
    """
    Strip of consecutive STIR futures.

    Useful for:
    - Building forward curves
    - Hedging swap exposures
    - Creating synthetic rate exposure
    """

    def __init__(
        self,
        contracts: List[STIRFuture],
        positions: Optional[List[int]] = None,
    ):
        """
        Initialize futures strip.

        Args:
            contracts: List of futures contracts
            positions: Number of contracts (+ = long, - = short)
        """
        self.contracts = sorted(contracts, key=lambda x: x.contract_date)
        self.positions = positions or [1] * len(contracts)

    @classmethod
    def quarterly_strip(
        cls,
        start_date: date,
        num_contracts: int,
        contract_type: FuturesContractType = FuturesContractType.SOFR_3M,
    ) -> "FuturesStrip":
        """
        Create quarterly futures strip.

        Args:
            start_date: First contract date
            num_contracts: Number of contracts
            contract_type: Type of futures

        Returns:
            FuturesStrip
        """
        contracts = []
        current_date = start_date

        for _ in range(num_contracts):
            fut = STIRFuture(
                contract_date=current_date,
                contract_type=contract_type,
            )
            contracts.append(fut)

            # Move to next quarter (IMM date convention)
            month = current_date.month
            if month in [3, 6, 9, 12]:
                next_month = (month % 12) + 3
                year_add = 1 if next_month <= 3 else 0
            else:
                next_month = ((month - 1) // 3 + 1) * 3
                year_add = 0

            current_date = date(
                current_date.year + year_add,
                next_month,
                15  # Third Wednesday approximation
            )

        return cls(contracts)

    def implied_rates(
        self,
        valuation_date: date,
        curve: Curve,
        volatility: float = 0.01,
    ) -> List[float]:
        """Get implied forward rates for all contracts."""
        return [
            c.forward_rate(curve) + c.convexity_adjustment(valuation_date, curve, volatility)
            for c in self.contracts
        ]

    def total_dv01(self) -> float:
        """Calculate total DV01 of strip."""
        def sign(x):
            if x > 0:
                return 1
            elif x < 0:
                return -1
            return 0
        return sum(c.dv01(abs(p)) * sign(p)
                   for c, p in zip(self.contracts, self.positions))

    def hedge_ratio_for_swap(
        self,
        swap_dv01: float,
    ) -> List[int]:
        """
        Calculate hedge ratios to match swap DV01.

        Simple approach: allocate proportionally across contracts.
        """
        contract_dv01s = [c.dv01() for c in self.contracts]
        total = sum(contract_dv01s)

        if total == 0:
            return [0] * len(self.contracts)

        # Proportion each contract's DV01 to match target
        ratios = []
        for dv01 in contract_dv01s:
            ratio = int(round(swap_dv01 / dv01))
            ratios.append(ratio)

        return ratios

    def __len__(self) -> int:
        return len(self.contracts)

    def __iter__(self):
        return iter(self.contracts)
