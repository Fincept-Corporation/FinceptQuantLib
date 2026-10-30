"""
SA-CCR (Standardized Approach for Counterparty Credit Risk)

Implements Basel III SA-CCR framework for calculating:
- Exposure at Default (EAD) for derivatives
- Replacement Cost (RC)
- Potential Future Exposure (PFE)

Used for CVA capital charge and CCR capital requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
import math


class AssetClassCCR(Enum):
    """SA-CCR asset classes."""
    INTEREST_RATE = "interest_rate"
    FOREIGN_EXCHANGE = "fx"
    CREDIT = "credit"
    EQUITY = "equity"
    COMMODITY = "commodity"


@dataclass
class HedgingSet:
    """
    Hedging set for netting derivatives.

    Groups trades by asset class and risk factors that can offset.
    """
    asset_class: AssetClassCCR
    risk_factor: str  # e.g., "USD-LIBOR-3M", "EUR/USD"
    trades: List[str]  # Trade IDs
    notionals: List[float]
    maturities: List[float]  # Years
    supervisory_deltas: List[float]

    def effective_notional(self) -> float:
        """Calculate effective notional for the hedging set."""
        # Weighted by maturity factor and delta
        # SA-CCR: maturity factor = sqrt(min(M, 1))
        total = 0.0
        for i, notional in enumerate(self.notionals):
            maturity = self.maturities[i] if i < len(self.maturities) else 1.0
            delta = self.supervisory_deltas[i] if i < len(self.supervisory_deltas) else 1.0
            mf = math.sqrt(min(maturity, 1.0))
            total += notional * mf * delta

        return abs(total)


@dataclass
class MaturityFactor:
    """Maturity factor for PFE calculation."""
    maturity: float  # Years

    def mf(self) -> float:
        """
        Calculate maturity factor.

        MF = sqrt(min(maturity, 1 year) / 1 year)
        """
        return math.sqrt(min(self.maturity, 1.0))


@dataclass
class SupervisoryDelta:
    """
    Supervisory delta for options.

    Captures moneyness and time to maturity effects.
    """
    strike: float
    underlying: float
    time_to_expiry: float  # Years
    is_call: bool = True
    is_long: bool = True

    def delta(self) -> float:
        """
        Calculate supervisory delta.

        Simplified formula from SA-CCR standards.
        """
        # Moneyness
        if self.underlying == 0:
            return 0.0

        if self.strike <= 0:
            raise ValueError(f"Strike must be positive, got {self.strike}")
        ln_strike_spot = math.log(self.strike / self.underlying)

        # Supervisory volatility (simplified - typically 50% for IR, 120% for FX, etc.)
        sigma = 0.50

        # Time-adjusted volatility
        sqrt_t = math.sqrt(max(self.time_to_expiry, 0.01))

        # Approximation of Black-Scholes delta
        from scipy.stats import norm
        d = -ln_strike_spot / (sigma * sqrt_t) + 0.5 * sigma * sqrt_t

        if self.is_call:
            delta = norm.cdf(d)
        else:
            delta = norm.cdf(d) - 1

        # Adjust for long/short
        return delta if self.is_long else -delta


@dataclass
class ReplacementCost:
    """
    Replacement cost component of EAD.

    RC = max(V - C, 0)
    where V = mark-to-market value, C = net collateral
    """
    mtm_value: float  # Mark-to-market of derivatives portfolio
    collateral_posted: float = 0.0
    collateral_received: float = 0.0

    def rc(self) -> float:
        """Calculate replacement cost."""
        net_collateral = self.collateral_received - self.collateral_posted
        return max(self.mtm_value - net_collateral, 0)


@dataclass
class PotentialFutureExposure:
    """
    Potential Future Exposure component.

    PFE = multiplier * AddOn
    """
    addon: float
    mtm_value: float
    collateral_amount: float = 0.0

    def multiplier(self) -> float:
        """
        Calculate multiplier.

        Multiplier = min(1, Floor + (1-Floor) * exp(V-C / (2*(1-Floor)*AddOn)))
        Floor = 0.05
        """
        floor = 0.05
        v_c = self.mtm_value - self.collateral_amount

        if self.addon == 0:
            return 1.0

        exp_term = math.exp(v_c / (2 * (1 - floor) * self.addon))
        mult = min(1.0, floor + (1 - floor) * exp_term)
        return mult

    def pfe(self) -> float:
        """Calculate PFE."""
        return self.multiplier() * self.addon


@dataclass
class ExposureAtDefaultCCR:
    """
    Complete EAD calculation for CCR.

    EAD = alpha * (RC + PFE)
    where alpha = 1.4 (supervisory parameter)
    """
    replacement_cost: ReplacementCost
    potential_future_exposure: PotentialFutureExposure
    alpha: float = 1.4

    def ead(self) -> float:
        """Calculate exposure at default."""
        rc = self.replacement_cost.rc()
        pfe = self.potential_future_exposure.pfe()
        return self.alpha * (rc + pfe)


class SACCRCalculator:
    """
    Complete SA-CCR calculator.

    Calculates counterparty credit risk exposure for derivatives portfolios.

    Examples:
        >>> saccr = SACCRCalculator()
        >>> saccr.add_hedging_set(HedgingSet(
        ...     AssetClassCCR.INTEREST_RATE,
        ...     "USD-SOFR",
        ...     ["SWAP1", "SWAP2"],
        ...     [10_000_000, -5_000_000],
        ...     [5.0, 3.0],
        ...     [1.0, -1.0]
        ... ))
        >>> ead = saccr.calculate_ead(mtm_value=50000, collateral=30000)
        >>> print(f"EAD: ${ead:,.2f}")
        EAD: $133,000.00
    """

    # Supervisory factors by asset class
    SUPERVISORY_FACTORS = {
        AssetClassCCR.INTEREST_RATE: 0.005,
        AssetClassCCR.FOREIGN_EXCHANGE: 0.04,
        AssetClassCCR.CREDIT: 0.005,
        AssetClassCCR.EQUITY: 0.32,
        AssetClassCCR.COMMODITY: 0.40,
    }

    def __init__(self) -> None:
        """Initialize SA-CCR calculator."""
        self.hedging_sets: List[HedgingSet] = []

    def add_hedging_set(self, hedging_set: HedgingSet) -> None:
        """Add a hedging set to the netting set."""
        self.hedging_sets.append(hedging_set)

    def calculate_addon(self) -> float:
        """
        Calculate aggregate add-on across all hedging sets.

        AddOn = Sum over asset classes of AddOn_class
        """
        addons_by_class: Dict[AssetClassCCR, float] = {}

        for hs in self.hedging_sets:
            if hs.asset_class not in addons_by_class:
                addons_by_class[hs.asset_class] = 0.0

            # Effective notional
            eff_notional = hs.effective_notional()

            # Supervisory factor
            sf = self.SUPERVISORY_FACTORS.get(hs.asset_class, 0.10)

            # Maturity factor (simplified)
            avg_maturity = sum(hs.maturities) / len(hs.maturities) if hs.maturities else 1.0
            mf = MaturityFactor(avg_maturity).mf()

            # Add-on = SF * MF * Effective Notional
            addon = sf * mf * eff_notional
            addons_by_class[hs.asset_class] += addon

        return sum(addons_by_class.values())

    def calculate_ead(
        self,
        mtm_value: float,
        collateral: float = 0.0
    ) -> float:
        """
        Calculate total EAD for the netting set.

        Args:
            mtm_value: Current mark-to-market value
            collateral: Net collateral amount

        Returns:
            Exposure at default
        """
        # Replacement Cost
        rc = ReplacementCost(mtm_value, collateral_received=collateral)

        # Add-on
        addon = self.calculate_addon()

        # PFE
        pfe = PotentialFutureExposure(addon, mtm_value, collateral)

        # EAD
        ead_ccr = ExposureAtDefaultCCR(rc, pfe)
        return ead_ccr.ead()


def calculate_ead_ccr(
    mtm_value: float,
    addon: float,
    collateral: float = 0.0,
    alpha: float = 1.4
) -> float:
    """
    Simple EAD calculation for CCR.

    Args:
        mtm_value: Mark-to-market value
        addon: Aggregate add-on amount
        collateral: Net collateral
        alpha: Alpha factor (default 1.4)

    Returns:
        Exposure at default
    """
    rc = max(mtm_value - collateral, 0)
    pfe = addon  # Simplified (ignores multiplier)
    return alpha * (rc + pfe)


def calculate_pfe(addon: float, mtm_value: float, collateral: float = 0.0) -> float:
    """Calculate potential future exposure."""
    pfe = PotentialFutureExposure(addon, mtm_value, collateral)
    return pfe.pfe()


def calculate_rc(mtm_value: float, collateral: float = 0.0) -> float:
    """Calculate replacement cost."""
    rc = ReplacementCost(mtm_value, collateral_received=collateral)
    return rc.rc()


def addon_aggregation(addons: List[float], correlations: Optional[List[List[float]]] = None) -> float:
    """
    Aggregate add-ons across hedging sets with correlations.

    Args:
        addons: List of add-on amounts
        correlations: Correlation matrix (if None, assumes perfect correlation)

    Returns:
        Aggregated add-on
    """
    if not addons:
        return 0.0

    if correlations is None:
        # Perfect correlation: simple sum
        return sum(addons)

    # With correlations: quadratic form
    total = 0.0
    for i, addon_i in enumerate(addons):
        for j, addon_j in enumerate(addons):
            rho = correlations[i][j] if i < len(correlations) and j < len(correlations[i]) else 1.0
            total += addon_i * addon_j * rho

    return math.sqrt(max(total, 0))
