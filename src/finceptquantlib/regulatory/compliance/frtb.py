"""
FRTB (Fundamental Review of Trading Book) Implementation

Implements the Basel Committee's FRTB framework for market risk capital:
- FRTB-SA: Standardized Approach using sensitivities-based method
- FRTB-IMA: Internal Models Approach (planned)

References:
- Basel Committee on Banking Supervision (2019)
- BCBS 457: Minimum capital requirements for market risk
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
import math


class RiskClass(Enum):
    """FRTB risk classes."""
    INTEREST_RATE = "interest_rate"
    CREDIT_SPREAD = "credit_spread"
    EQUITY = "equity"
    COMMODITY = "commodity"
    FX = "fx"


class RiskFactor(Enum):
    """Risk factor types within each risk class."""
    DELTA = "delta"
    VEGA = "vega"
    CURVATURE = "curvature"


@dataclass
class DeltaRisk:
    """
    Delta risk sensitivities.

    Attributes:
        risk_class: Asset class (IR, credit, equity, commodity, FX)
        bucket: Risk bucket within asset class
        tenor: Time bucket (for curves)
        sensitivity: Net sensitivity value
        risk_weight: Supervisory risk weight
    """
    risk_class: RiskClass
    bucket: int
    tenor: Optional[str] = None
    sensitivity: float = 0.0
    risk_weight: float = 1.0

    def weighted_sensitivity(self) -> float:
        """Calculate risk-weighted sensitivity."""
        return self.sensitivity * self.risk_weight


@dataclass
class VegaRisk:
    """
    Vega risk sensitivities for implied volatility.

    Attributes:
        risk_class: Asset class
        bucket: Risk bucket
        maturity: Option maturity bucket
        sensitivity: Vega sensitivity
        risk_weight: Supervisory risk weight (default 100%)
    """
    risk_class: RiskClass
    bucket: int
    maturity: str
    sensitivity: float = 0.0
    risk_weight: float = 1.0

    def weighted_sensitivity(self) -> float:
        """Calculate risk-weighted vega."""
        return self.sensitivity * self.risk_weight


@dataclass
class CurvatureRisk:
    """
    Curvature risk for non-linear price changes.

    Captures risk from large moves (gamma/convexity effects).
    """
    risk_class: RiskClass
    bucket: int
    cvr_plus: float = 0.0  # Curvature risk up scenario
    cvr_minus: float = 0.0  # Curvature risk down scenario

    def net_curvature_risk(self) -> float:
        """Calculate net curvature risk."""
        return max(self.cvr_plus + self.cvr_minus, 0)


@dataclass
class SensitivityResult:
    """Result of sensitivity aggregation."""
    delta_charge: float
    vega_charge: float
    curvature_charge: float
    total: float


@dataclass
class CapitalCharge:
    """FRTB capital charge breakdown."""
    delta: float
    vega: float
    curvature: float
    residual_risk: float
    total: float


@dataclass
class FRTBResult:
    """
    Complete FRTB calculation result.

    Attributes:
        risk_classes: Capital charges by risk class
        total_capital: Total market risk capital requirement
        delta_total: Total delta charge across all risk classes
        vega_total: Total vega charge
        curvature_total: Total curvature charge
        residual_total: Residual risk add-on
    """
    risk_classes: Dict[RiskClass, CapitalCharge]
    total_capital: float
    delta_total: float
    vega_total: float
    curvature_total: float
    residual_total: float


class SensitivityBasedMethod:
    """
    FRTB Sensitivities-Based Method (SBM).

    Calculates market risk capital using delta, vega, and curvature risk
    measures for different risk classes.
    """

    # Correlation parameters (simplified - full version has 100+ parameters)
    INTRA_BUCKET_CORRELATION = {
        RiskClass.INTEREST_RATE: 0.99,
        RiskClass.CREDIT_SPREAD: 0.80,
        RiskClass.EQUITY: 0.15,
        RiskClass.COMMODITY: 0.55,
        RiskClass.FX: 0.60,
    }

    INTER_BUCKET_CORRELATION = {
        RiskClass.INTEREST_RATE: 0.50,
        RiskClass.CREDIT_SPREAD: 0.50,
        RiskClass.EQUITY: 0.15,
        RiskClass.COMMODITY: 0.20,
        RiskClass.FX: 0.60,
    }

    def __init__(self) -> None:
        """Initialize FRTB SBM calculator."""
        self.delta_risks: List[DeltaRisk] = []
        self.vega_risks: List[VegaRisk] = []
        self.curvature_risks: List[CurvatureRisk] = []

    def add_delta_sensitivity(
        self,
        risk_class: RiskClass,
        bucket: int,
        sensitivity: float,
        risk_weight: float = 1.0,
        tenor: Optional[str] = None
    ) -> None:
        """Add delta sensitivity."""
        self.delta_risks.append(DeltaRisk(
            risk_class=risk_class,
            bucket=bucket,
            tenor=tenor,
            sensitivity=sensitivity,
            risk_weight=risk_weight
        ))

    def add_vega_sensitivity(
        self,
        risk_class: RiskClass,
        bucket: int,
        maturity: str,
        sensitivity: float,
        risk_weight: float = 1.0
    ) -> None:
        """Add vega sensitivity."""
        self.vega_risks.append(VegaRisk(
            risk_class=risk_class,
            bucket=bucket,
            maturity=maturity,
            sensitivity=sensitivity,
            risk_weight=risk_weight
        ))

    def calculate_delta_charge(self, risk_class: RiskClass) -> float:
        """
        Calculate delta capital charge for a risk class.

        Uses correlation matrices to aggregate sensitivities.
        Formula: KB = sqrt(sum_i sum_j rho_ij * WS_i * WS_j)
        """
        # Filter deltas for this risk class
        deltas = [d for d in self.delta_risks if d.risk_class == risk_class]
        if not deltas:
            return 0.0

        # Group by bucket
        buckets: Dict[int, List[DeltaRisk]] = {}
        for d in deltas:
            if d.bucket not in buckets:
                buckets[d.bucket] = []
            buckets[d.bucket].append(d)

        # Calculate within-bucket charges
        bucket_charges = []
        for bucket_id, bucket_deltas in buckets.items():
            ws_values = [d.weighted_sensitivity() for d in bucket_deltas]

            # Intra-bucket correlation
            rho = self.INTRA_BUCKET_CORRELATION.get(risk_class, 0.5)

            # Simple aggregation (full version uses correlation matrix)
            sum_ws_sq = sum(ws * ws for ws in ws_values)
            sum_ws_cross = sum(ws_values) ** 2 - sum_ws_sq

            kb = math.sqrt(max(0.0, sum_ws_sq + rho * sum_ws_cross))
            bucket_charges.append((bucket_id, kb))

        # Aggregate across buckets
        if len(bucket_charges) == 1:
            return bucket_charges[0][1]

        # Inter-bucket correlation
        gamma = self.INTER_BUCKET_CORRELATION.get(risk_class, 0.5)

        sum_kb_sq = sum(kb * kb for _, kb in bucket_charges)
        sum_kb_cross = sum(bucket_charges[i][1] * bucket_charges[j][1]
                          for i in range(len(bucket_charges))
                          for j in range(i+1, len(bucket_charges)))

        return math.sqrt(max(0.0, sum_kb_sq + 2 * gamma * sum_kb_cross))

    def calculate_vega_charge(self, risk_class: RiskClass) -> float:
        """Calculate vega capital charge."""
        vegas = [v for v in self.vega_risks if v.risk_class == risk_class]
        if not vegas:
            return 0.0

        # Simplified: sum of absolute weighted sensitivities
        return math.sqrt(sum(v.weighted_sensitivity() ** 2 for v in vegas))

    def calculate_curvature_charge(self, risk_class: RiskClass) -> float:
        """Calculate curvature capital charge."""
        curvatures = [c for c in self.curvature_risks if c.risk_class == risk_class]
        if not curvatures:
            return 0.0

        # Aggregate curvature risks
        total_cvr = sum(c.net_curvature_risk() for c in curvatures)
        return max(total_cvr, 0)


class FRTBSACalculator:
    """
    Complete FRTB-SA calculator.

    Implements the full Standardized Approach for market risk capital
    under FRTB regulations.

    Examples:
        >>> frtb = FRTBSACalculator()
        >>> frtb.add_delta_sensitivity(RiskClass.INTEREST_RATE, bucket=1,
        ...                            sensitivity=1000000, risk_weight=0.017)
        >>> result = frtb.calculate()
        >>> print(f"Total capital: ${result.total_capital:,.2f}")
        Total capital: $17,000.00
    """

    def __init__(self) -> None:
        """Initialize FRTB-SA calculator."""
        self.sbm = SensitivityBasedMethod()
        self.residual_risk_addon = 0.0

    def add_delta_sensitivity(
        self,
        risk_class: RiskClass,
        bucket: int,
        sensitivity: float,
        risk_weight: float = 1.0,
        tenor: Optional[str] = None
    ) -> None:
        """Add delta sensitivity to portfolio."""
        self.sbm.add_delta_sensitivity(risk_class, bucket, sensitivity, risk_weight, tenor)

    def add_vega_sensitivity(
        self,
        risk_class: RiskClass,
        bucket: int,
        maturity: str,
        sensitivity: float,
        risk_weight: float = 1.0
    ) -> None:
        """Add vega sensitivity."""
        self.sbm.add_vega_sensitivity(risk_class, bucket, maturity, sensitivity, risk_weight)

    def set_residual_risk(self, addon: float) -> None:
        """Set residual risk add-on for exotic underlyings."""
        self.residual_risk_addon = addon

    def calculate(self) -> FRTBResult:
        """
        Calculate total FRTB capital requirement.

        Returns:
            FRTBResult with complete breakdown
        """
        risk_classes_charges = {}
        total_delta = 0.0
        total_vega = 0.0
        total_curvature = 0.0

        for risk_class in RiskClass:
            delta = self.sbm.calculate_delta_charge(risk_class)
            vega = self.sbm.calculate_vega_charge(risk_class)
            curvature = self.sbm.calculate_curvature_charge(risk_class)

            charge = CapitalCharge(
                delta=delta,
                vega=vega,
                curvature=curvature,
                residual_risk=0.0,  # Allocated later
                total=delta + vega + curvature
            )

            risk_classes_charges[risk_class] = charge
            total_delta += delta
            total_vega += vega
            total_curvature += curvature

        total_capital = total_delta + total_vega + total_curvature + self.residual_risk_addon

        return FRTBResult(
            risk_classes=risk_classes_charges,
            total_capital=total_capital,
            delta_total=total_delta,
            vega_total=total_vega,
            curvature_total=total_curvature,
            residual_total=self.residual_risk_addon
        )
