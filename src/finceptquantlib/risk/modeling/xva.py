"""
XVA - Valuation Adjustments

Provides:
- CVA (Credit Valuation Adjustment)
- DVA (Debit Valuation Adjustment)
- FVA (Funding Valuation Adjustment)
- MVA (Margin Valuation Adjustment)
- KVA (Capital Valuation Adjustment)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Callable
import math
import random


@dataclass
class XVAResult:
    """Result of XVA calculation."""
    cva: float
    dva: float
    fva: float
    mva: float
    kva: float
    total_xva: float
    details: dict


@dataclass
class CreditParameters:
    """Credit parameters for XVA."""
    counterparty_spread: float  # CDS spread (bps)
    counterparty_recovery: float = 0.4
    own_spread: float = 0.0  # Own CDS spread for DVA
    own_recovery: float = 0.4


@dataclass
class FundingParameters:
    """Funding parameters for FVA."""
    funding_spread: float  # Spread over risk-free (bps)
    lending_spread: float = 0.0  # What we earn on collateral


class CVACalculator:
    """
    Credit Valuation Adjustment calculator.

    CVA = (1-R) * integral(0 to T) of EE(t) * dPD(t)

    where:
    - R = Recovery rate
    - EE(t) = Expected Exposure at time t
    - PD(t) = Probability of default by time t
    """

    def __init__(
        self,
        credit_params: CreditParameters,
        time_grid: Optional[list[float]] = None,
    ):
        self.credit_params = credit_params
        self._time_grid = time_grid

    def calculate_cva(
        self,
        expected_exposures: list[float],
        time_points: list[float],
        discount_factors: list[float],
    ) -> float:
        """
        Calculate CVA from expected exposure profile.

        Args:
            expected_exposures: EE at each time point
            time_points: Time points (years)
            discount_factors: Discount factors at each time

        Returns:
            CVA value
        """
        lgd = 1 - self.credit_params.counterparty_recovery
        spread = self.credit_params.counterparty_spread / 10000  # Convert bps to decimal

        # Hazard rate from spread (assuming constant)
        hazard_rate = spread / lgd

        cva = 0.0
        for i in range(1, len(time_points)):
            t_prev = time_points[i - 1]
            t = time_points[i]

            # Average EE over period
            avg_ee = (expected_exposures[i - 1] + expected_exposures[i]) / 2

            # Survival probabilities
            surv_prev = math.exp(-hazard_rate * t_prev)
            surv = math.exp(-hazard_rate * t)

            # Default probability in period
            default_prob = surv_prev - surv

            # Discount factor at midpoint
            avg_df = (discount_factors[i - 1] + discount_factors[i]) / 2

            cva += lgd * avg_ee * default_prob * avg_df

        return cva

    def calculate_dva(
        self,
        expected_negative_exposures: list[float],
        time_points: list[float],
        discount_factors: list[float],
    ) -> float:
        """
        Calculate DVA from expected negative exposure profile.

        DVA is similar to CVA but from own default perspective.
        """
        lgd = 1 - self.credit_params.own_recovery
        spread = self.credit_params.own_spread / 10000

        if spread <= 0:
            return 0.0

        hazard_rate = spread / lgd

        dva = 0.0
        for i in range(1, len(time_points)):
            t_prev = time_points[i - 1]
            t = time_points[i]

            # ENE is negative of our exposure (counterparty's exposure to us)
            avg_ene = (expected_negative_exposures[i - 1] + expected_negative_exposures[i]) / 2

            surv_prev = math.exp(-hazard_rate * t_prev)
            surv = math.exp(-hazard_rate * t)
            default_prob = surv_prev - surv

            avg_df = (discount_factors[i - 1] + discount_factors[i]) / 2

            dva += lgd * avg_ene * default_prob * avg_df

        return dva


class FVACalculator:
    """
    Funding Valuation Adjustment calculator.

    FVA = integral of (EE(t) * s_f - ENE(t) * s_l) * DF(t) dt

    where:
    - s_f = funding spread
    - s_l = lending spread
    """

    def __init__(
        self,
        funding_params: FundingParameters,
    ):
        self.funding_params = funding_params

    def calculate_fva(
        self,
        expected_exposures: list[float],
        expected_negative_exposures: list[float],
        time_points: list[float],
        discount_factors: list[float],
    ) -> float:
        """
        Calculate FVA.

        Args:
            expected_exposures: Positive exposures (we need to fund)
            expected_negative_exposures: Negative exposures (we receive collateral)
            time_points: Time grid
            discount_factors: Discount factors

        Returns:
            FVA value
        """
        funding_spread = self.funding_params.funding_spread / 10000
        lending_spread = self.funding_params.lending_spread / 10000

        fva = 0.0
        for i in range(1, len(time_points)):
            t_prev = time_points[i - 1]
            t = time_points[i]
            dt = t - t_prev

            avg_ee = (expected_exposures[i - 1] + expected_exposures[i]) / 2
            avg_ene = (expected_negative_exposures[i - 1] + expected_negative_exposures[i]) / 2
            avg_df = (discount_factors[i - 1] + discount_factors[i]) / 2

            # Funding cost on positive exposure
            fva += avg_ee * funding_spread * dt * avg_df

            # Funding benefit on negative exposure
            fva -= avg_ene * lending_spread * dt * avg_df

        return fva


class MVACalculator:
    """
    Margin Valuation Adjustment calculator.

    MVA represents the cost of posting initial margin.

    MVA = integral of IM(t) * s_f * DF(t) dt
    """

    def __init__(
        self,
        funding_spread: float,  # in bps
    ):
        self.funding_spread = funding_spread / 10000

    def calculate_mva(
        self,
        initial_margins: list[float],
        time_points: list[float],
        discount_factors: list[float],
    ) -> float:
        """
        Calculate MVA.

        Args:
            initial_margins: IM profile over time
            time_points: Time grid
            discount_factors: Discount factors

        Returns:
            MVA value
        """
        mva = 0.0
        for i in range(1, len(time_points)):
            t_prev = time_points[i - 1]
            t = time_points[i]
            dt = t - t_prev

            avg_im = (initial_margins[i - 1] + initial_margins[i]) / 2
            avg_df = (discount_factors[i - 1] + discount_factors[i]) / 2

            mva += avg_im * self.funding_spread * dt * avg_df

        return mva


class KVACalculator:
    """
    Capital Valuation Adjustment calculator.

    KVA represents the cost of regulatory capital.

    KVA = integral of K(t) * r_hurdle * DF(t) dt

    where K(t) is regulatory capital requirement.
    """

    def __init__(
        self,
        hurdle_rate: float,  # Required return on capital
        capital_percentage: float = 0.08,  # Basel capital ratio
    ):
        self.hurdle_rate = hurdle_rate
        self.capital_percentage = capital_percentage

    def calculate_kva(
        self,
        rwas: list[float],  # Risk-weighted assets over time
        time_points: list[float],
        discount_factors: list[float],
    ) -> float:
        """
        Calculate KVA.

        Args:
            rwas: RWA profile over time
            time_points: Time grid
            discount_factors: Discount factors

        Returns:
            KVA value
        """
        kva = 0.0
        for i in range(1, len(time_points)):
            t_prev = time_points[i - 1]
            t = time_points[i]
            dt = t - t_prev

            # Capital = RWA * capital ratio
            avg_rwa = (rwas[i - 1] + rwas[i]) / 2
            avg_capital = avg_rwa * self.capital_percentage

            avg_df = (discount_factors[i - 1] + discount_factors[i]) / 2

            kva += avg_capital * self.hurdle_rate * dt * avg_df

        return kva


class XVAEngine:
    """
    Comprehensive XVA calculation engine.

    Combines all valuation adjustments.
    """

    def __init__(
        self,
        credit_params: CreditParameters,
        funding_params: FundingParameters,
        hurdle_rate: float = 0.10,
    ):
        self.cva_calc = CVACalculator(credit_params)
        self.fva_calc = FVACalculator(funding_params)
        self.mva_calc = MVACalculator(funding_params.funding_spread)
        self.kva_calc = KVACalculator(hurdle_rate)

    def calculate_all(
        self,
        expected_exposures: list[float],
        expected_negative_exposures: list[float],
        initial_margins: list[float],
        rwas: list[float],
        time_points: list[float],
        discount_factors: list[float],
    ) -> XVAResult:
        """
        Calculate all XVA components.

        Args:
            expected_exposures: EE profile
            expected_negative_exposures: ENE profile
            initial_margins: IM profile
            rwas: RWA profile
            time_points: Time grid
            discount_factors: DF profile

        Returns:
            XVAResult with all components
        """
        cva = self.cva_calc.calculate_cva(
            expected_exposures, time_points, discount_factors
        )

        dva = self.cva_calc.calculate_dva(
            expected_negative_exposures, time_points, discount_factors
        )

        fva = self.fva_calc.calculate_fva(
            expected_exposures, expected_negative_exposures,
            time_points, discount_factors
        )

        mva = self.mva_calc.calculate_mva(
            initial_margins, time_points, discount_factors
        )

        kva = self.kva_calc.calculate_kva(
            rwas, time_points, discount_factors
        )

        total = cva - dva + fva + mva + kva

        return XVAResult(
            cva=cva,
            dva=dva,
            fva=fva,
            mva=mva,
            kva=kva,
            total_xva=total,
            details={
                "bilateral_cva": cva - dva,
                "funding_costs": fva,
                "margin_costs": mva,
                "capital_costs": kva,
            },
        )


def simulate_exposure_profile(
    pricing_func: Callable[[date, float], float],
    valuation_date: date,
    maturity_years: float,
    n_paths: int = 1000,
    n_steps: int = 100,
    volatility: float = 0.01,
) -> tuple:
    """
    Simulate exposure profile using Monte Carlo.

    Args:
        pricing_func: Function(date, market_rate) -> NPV
        valuation_date: Start date
        maturity_years: Time to maturity
        n_paths: Number of simulation paths
        n_steps: Number of time steps
        volatility: Market rate volatility

    Returns:
        (time_points, expected_exposures, expected_negative_exposures)
    """
    dt = maturity_years / n_steps
    time_points = [maturity_years * i / n_steps for i in range(n_steps + 1)]

    # Initial market state (e.g., rate = 0.03)
    initial_rate = 0.03

    # Simulate paths
    all_exposures = [[0.0] * (n_steps + 1) for _ in range(n_paths)]

    for path in range(n_paths):
        rate = initial_rate
        for step in range(n_steps + 1):
            # Calculate NPV at this point
            sim_date = valuation_date  # Simplified
            npv = pricing_func(sim_date, rate)
            all_exposures[path][step] = npv

            if step < n_steps:
                # Evolve rate (simple Brownian motion)
                dW = random.gauss(0, math.sqrt(dt))
                rate = rate + volatility * dW

    # Calculate expected exposures
    expected_exposures = [0.0] * (n_steps + 1)
    expected_negative_exposures = [0.0] * (n_steps + 1)

    for step in range(n_steps + 1):
        pos_sum = 0.0
        neg_sum = 0.0
        for path in range(n_paths):
            exp = all_exposures[path][step]
            pos_sum += max(exp, 0)
            neg_sum += max(-exp, 0)
        expected_exposures[step] = pos_sum / n_paths
        expected_negative_exposures[step] = neg_sum / n_paths

    return time_points, expected_exposures, expected_negative_exposures


def calculate_potential_future_exposure(
    exposure_paths: list[list[float]],
    confidence_level: float = 0.95,
) -> list[float]:
    """
    Calculate Potential Future Exposure (PFE) at given confidence level.

    PFE(t) = percentile of exposure distribution at time t

    Args:
        exposure_paths: Shape (n_paths, n_times)
        confidence_level: Percentile (e.g., 0.95 for 95%)

    Returns:
        PFE profile over time
    """
    n_paths = len(exposure_paths)
    n_times = len(exposure_paths[0]) if exposure_paths else 0
    pfe = []

    for t in range(n_times):
        # Get positive exposures at time t
        pos_exposures = sorted([max(exposure_paths[p][t], 0) for p in range(n_paths)])
        # Percentile
        idx = int(confidence_level * n_paths)
        idx = min(idx, n_paths - 1)
        pfe.append(pos_exposures[idx])

    return pfe


def calculate_expected_positive_exposure(
    exposure_paths: list[list[float]],
) -> list[float]:
    """
    Calculate Expected Positive Exposure (EPE).

    EPE(t) = E[max(V(t), 0)]
    """
    n_paths = len(exposure_paths)
    n_times = len(exposure_paths[0]) if exposure_paths else 0
    epe = []

    for t in range(n_times):
        pos_sum = sum(max(exposure_paths[p][t], 0) for p in range(n_paths))
        epe.append(pos_sum / n_paths if n_paths > 0 else 0.0)

    return epe
