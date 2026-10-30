"""
Basis Swaps

Single currency basis swaps that exchange floating rates
based on different indices.

Types:
- Tenor Basis Swap: 3M vs 6M EURIBOR
- Index Basis Swap: SOFR vs Fed Funds
- RFR Basis Swap: Compounded RFR vs Term Rate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional, TYPE_CHECKING

from finceptquantlib.core.foundations.types import (
    Currency, Rate
)
from finceptquantlib.core.instruments.periods import (
    ObservationMethod
)
from finceptquantlib.core.instruments.legs import (
    FloatLeg, CompoundingLeg
)

if TYPE_CHECKING:
    from finceptquantlib.curves.curve import Curve


# =============================================================================
# Single Currency Basis Swap
# =============================================================================

@dataclass
class SingleCurrencyBasisSwap:
    """
    Single Currency Basis Swap (SBS).

    Exchanges two floating rate streams based on different indices or tenors.

    Common types:
    - SOFR vs Fed Funds
    - 3M EURIBOR vs 6M EURIBOR
    - Term SOFR vs Compounded SOFR

    Args:
        effective_date: Start date
        termination_date: End date
        notional: Notional amount
        currency: Currency
        pay_index: Index for pay leg
        receive_index: Index for receive leg
        pay_frequency: Payment frequency for pay leg
        receive_frequency: Payment frequency for receive leg
        spread: Spread on pay leg (convention)
        spread_on_receive: If True, spread is on receive leg
    """
    effective_date: date
    termination_date: date
    notional: float
    currency: Currency
    pay_index: str
    receive_index: str
    pay_frequency: str = "3M"
    receive_frequency: str = "6M"
    spread: Rate = field(default_factory=lambda: Rate(0.0))
    spread_on_receive: bool = False
    day_count_pay: str = "ACT/360"
    day_count_receive: str = "ACT/360"

    _pay_leg: Optional[FloatLeg] = field(default=None, repr=False)
    _receive_leg: Optional[FloatLeg] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.spread, (float, int, str)):
            self.spread = Rate(self.spread)
        if isinstance(self.currency, str):
            self.currency = Currency.from_string(self.currency)

        self._build_legs()

    def _build_legs(self) -> None:
        """Build the basis swap legs."""
        pay_spread = self.spread if not self.spread_on_receive else Rate(0.0)
        receive_spread = self.spread if self.spread_on_receive else Rate(0.0)

        self._pay_leg = FloatLeg(
            effective_date=self.effective_date,
            termination_date=self.termination_date,
            notional_amount=self.notional,
            ccy=self.currency,
            spread=pay_spread,
            index_name=self.pay_index,
            frequency=self.pay_frequency,
            day_count=self.day_count_pay,
            pay_receive=-1
        )

        self._receive_leg = FloatLeg(
            effective_date=self.effective_date,
            termination_date=self.termination_date,
            notional_amount=self.notional,
            ccy=self.currency,
            spread=receive_spread,
            index_name=self.receive_index,
            frequency=self.receive_frequency,
            day_count=self.day_count_receive,
            pay_receive=1
        )

    @property
    def pay_leg(self) -> FloatLeg:
        return self._pay_leg

    @property
    def receive_leg(self) -> FloatLeg:
        return self._receive_leg

    @property
    def tenor(self) -> str:
        """Swap tenor."""
        days = (self.termination_date - self.effective_date).days
        years = days / 365.0
        if years >= 1:
            return f"{int(round(years))}Y"
        return f"{int(round(years * 12))}M"

    def npv(
        self,
        discount_curve: Curve,
        pay_forecast_curve: Optional[Curve] = None,
        receive_forecast_curve: Optional[Curve] = None
    ) -> float:
        """
        Calculate NPV.

        Args:
            discount_curve: Discounting curve
            pay_forecast_curve: Forecast curve for pay leg index
            receive_forecast_curve: Forecast curve for receive leg index

        Returns:
            Net present value
        """
        pay_fc = pay_forecast_curve if pay_forecast_curve else discount_curve
        receive_fc = receive_forecast_curve if receive_forecast_curve else discount_curve

        pay_npv = self._pay_leg.npv(discount_curve, pay_fc)
        receive_npv = self._receive_leg.npv(discount_curve, receive_fc)

        return pay_npv + receive_npv

    def par_spread(
        self,
        discount_curve: Curve,
        pay_forecast_curve: Optional[Curve] = None,
        receive_forecast_curve: Optional[Curve] = None
    ) -> float:
        """
        Calculate par spread.

        The spread that makes NPV = 0.

        Returns:
            Par spread as decimal
        """
        pay_fc = pay_forecast_curve if pay_forecast_curve else discount_curve
        receive_fc = receive_forecast_curve if receive_forecast_curve else discount_curve

        # Create zero-spread swap
        zero_spread_swap = SingleCurrencyBasisSwap(
            effective_date=self.effective_date,
            termination_date=self.termination_date,
            notional=self.notional,
            currency=self.currency,
            pay_index=self.pay_index,
            receive_index=self.receive_index,
            pay_frequency=self.pay_frequency,
            receive_frequency=self.receive_frequency,
            spread=Rate(0.0),
            spread_on_receive=self.spread_on_receive
        )

        # NPV without spread
        base_npv = zero_spread_swap.npv(discount_curve, pay_fc, receive_fc)

        # Get spread leg annuity
        if self.spread_on_receive:
            spread_leg = zero_spread_swap.receive_leg
        else:
            spread_leg = zero_spread_swap.pay_leg

        annuity = spread_leg.spread_dv01(discount_curve, pay_fc) / 0.0001

        if abs(annuity) < 1e-10:
            return 0.0

        # Par spread = -NPV / Annuity (adjusting for sign convention)
        return -base_npv / annuity if not self.spread_on_receive else base_npv / annuity

    def basis_dv01(
        self,
        discount_curve: Curve,
        pay_forecast_curve: Optional[Curve] = None,
        receive_forecast_curve: Optional[Curve] = None
    ) -> float:
        """Calculate basis spread DV01."""
        pay_fc = pay_forecast_curve if pay_forecast_curve else discount_curve
        receive_fc = receive_forecast_curve if receive_forecast_curve else discount_curve

        base_npv = self.npv(discount_curve, pay_fc, receive_fc)

        # Bump spread
        bumped_spread = Rate(self.spread.decimal + 0.0001)
        bumped_swap = SingleCurrencyBasisSwap(
            effective_date=self.effective_date,
            termination_date=self.termination_date,
            notional=self.notional,
            currency=self.currency,
            pay_index=self.pay_index,
            receive_index=self.receive_index,
            pay_frequency=self.pay_frequency,
            receive_frequency=self.receive_frequency,
            spread=bumped_spread,
            spread_on_receive=self.spread_on_receive
        )
        bumped_npv = bumped_swap.npv(discount_curve, pay_fc, receive_fc)

        return bumped_npv - base_npv

    def __repr__(self) -> str:
        spread_str = f"+{self.spread}" if self.spread.decimal != 0 else ""
        return (
            f"SingleCurrencyBasisSwap("
            f"Pay {self.pay_index}/{self.pay_frequency}, "
            f"Receive {self.receive_index}/{self.receive_frequency}{spread_str}, "
            f"{self.tenor})"
        )


# =============================================================================
# Tenor Basis Swap
# =============================================================================

@dataclass
class TenorBasisSwap(SingleCurrencyBasisSwap):
    """
    Tenor Basis Swap.

    Exchanges two floating streams with different tenors of the same index.
    E.g., 3M EURIBOR vs 6M EURIBOR

    The spread compensates for:
    - Credit risk differential between tenors
    - Liquidity differential
    - Convexity adjustment
    """

    index_name: str = "EURIBOR"
    pay_tenor: str = "3M"
    receive_tenor: str = "6M"

    def __post_init__(self) -> None:
        # Set index names based on base index and tenors
        self.pay_index = f"{self.index_name}_{self.pay_tenor}"
        self.receive_index = f"{self.index_name}_{self.receive_tenor}"
        self.pay_frequency = self.pay_tenor
        self.receive_frequency = self.receive_tenor
        super().__post_init__()

    def __repr__(self) -> str:
        spread_str = f"+{self.spread}" if self.spread.decimal != 0 else ""
        return (
            f"TenorBasisSwap({self.index_name} "
            f"{self.pay_tenor} vs {self.receive_tenor}{spread_str}, "
            f"{self.tenor})"
        )


# =============================================================================
# RFR Basis Swap (Compounded vs Term)
# =============================================================================

@dataclass
class RFRBasisSwap:
    """
    RFR Basis Swap.

    Exchanges compounded RFR against term rate (e.g., Term SOFR).

    Pay: Compounded SOFR (in arrears)
    Receive: Term SOFR + spread

    Args:
        effective_date: Start date
        termination_date: End date
        notional: Notional amount
        currency: Currency
        rfr_index: Compounded RFR index name
        term_index: Term rate index name
        frequency: Payment frequency (both legs)
        spread: Spread on term leg
        observation_method: RFR observation method
        lookback_days: Lookback for RFR observation
        payment_delay: Payment delay for RFR leg
    """
    effective_date: date
    termination_date: date
    notional: float
    currency: Currency
    rfr_index: str = "SOFR"
    term_index: str = "TERM_SOFR"
    frequency: str = "3M"
    spread: Rate = field(default_factory=lambda: Rate(0.0))
    observation_method: ObservationMethod = ObservationMethod.PAYMENT_DELAY
    lookback_days: int = 2
    payment_delay: int = 2
    day_count: str = "ACT/360"

    _rfr_leg: Optional[CompoundingLeg] = field(default=None, repr=False)
    _term_leg: Optional[FloatLeg] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.spread, (float, int, str)):
            self.spread = Rate(self.spread)
        if isinstance(self.currency, str):
            self.currency = Currency.from_string(self.currency)

        self._build_legs()

    def _build_legs(self) -> None:
        """Build RFR and term legs."""
        # Compounded RFR leg (pay)
        self._rfr_leg = CompoundingLeg(
            effective_date=self.effective_date,
            termination_date=self.termination_date,
            notional_amount=self.notional,
            ccy=self.currency,
            spread=Rate(0.0),
            index_name=self.rfr_index,
            frequency=self.frequency,
            day_count=self.day_count,
            observation_method=self.observation_method,
            lookback_days=self.lookback_days,
            payment_delay=self.payment_delay,
            pay_receive=-1
        )

        # Term rate leg (receive)
        self._term_leg = FloatLeg(
            effective_date=self.effective_date,
            termination_date=self.termination_date,
            notional_amount=self.notional,
            ccy=self.currency,
            spread=self.spread,
            index_name=self.term_index,
            frequency=self.frequency,
            day_count=self.day_count,
            pay_receive=1
        )

    @property
    def rfr_leg(self) -> CompoundingLeg:
        return self._rfr_leg

    @property
    def term_leg(self) -> FloatLeg:
        return self._term_leg

    def npv(
        self,
        discount_curve: Curve,
        rfr_curve: Optional[Curve] = None,
        term_curve: Optional[Curve] = None
    ) -> float:
        """Calculate NPV."""
        rfr_fc = rfr_curve if rfr_curve else discount_curve
        term_fc = term_curve if term_curve else discount_curve

        rfr_npv = self._rfr_leg.npv(discount_curve, rfr_fc)
        term_npv = self._term_leg.npv(discount_curve, term_fc)

        return rfr_npv + term_npv

    def par_spread(
        self,
        discount_curve: Curve,
        rfr_curve: Optional[Curve] = None,
        term_curve: Optional[Curve] = None
    ) -> float:
        """Calculate par spread on term leg."""
        rfr_fc = rfr_curve if rfr_curve else discount_curve
        term_fc = term_curve if term_curve else discount_curve

        # Zero spread NPV
        zero_swap = RFRBasisSwap(
            effective_date=self.effective_date,
            termination_date=self.termination_date,
            notional=self.notional,
            currency=self.currency,
            rfr_index=self.rfr_index,
            term_index=self.term_index,
            frequency=self.frequency,
            spread=Rate(0.0),
            observation_method=self.observation_method
        )
        base_npv = zero_swap.npv(discount_curve, rfr_fc, term_fc)

        # Term leg annuity
        annuity = zero_swap.term_leg.spread_dv01(discount_curve, term_fc) / 0.0001

        if abs(annuity) < 1e-10:
            return 0.0

        return base_npv / annuity

    def __repr__(self) -> str:
        spread_str = f"+{self.spread}" if self.spread.decimal != 0 else ""
        tenor = f"{(self.termination_date - self.effective_date).days // 365}Y"
        return (
            f"RFRBasisSwap(Pay {self.rfr_index}, "
            f"Receive {self.term_index}{spread_str}, {tenor})"
        )


# =============================================================================
# Fed Funds vs SOFR Basis
# =============================================================================

@dataclass
class FFvSOFRBasis(SingleCurrencyBasisSwap):
    """
    Fed Funds vs SOFR Basis Swap.

    Standard USD basis swap trading Fed Funds against SOFR.
    """

    def __post_init__(self) -> None:
        self.currency = Currency.USD
        self.pay_index = "FED_FUNDS"
        self.receive_index = "SOFR"
        self.day_count_pay = "ACT/360"
        self.day_count_receive = "ACT/360"
        super().__post_init__()

    def __repr__(self) -> str:
        spread_str = f"+{self.spread}" if self.spread.decimal != 0 else ""
        return f"FFvSOFRBasis({self.tenor}{spread_str})"
