"""
Curve Bootstrapping and Calibration

Provides tools for:
- Single-curve bootstrapping
- Multi-curve calibration
- OIS discounting
- Spread curve construction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Union, Optional, List, Dict, Any
from enum import Enum
import math

from finceptquantlib.curves.curve import Curve, DiscountCurve
from finceptquantlib.solver.leastsq import LevenbergMarquardt


class InstrumentType(Enum):
    """Types of calibration instruments."""
    DEPOSIT = "deposit"
    FRA = "fra"
    FUTURE = "future"
    SWAP = "swap"
    OIS = "ois"
    BASIS_SWAP = "basis_swap"
    XCCY_SWAP = "xccy_swap"


@dataclass
class CalibrationInstrument:
    """
    A market instrument used for curve calibration.

    Attributes:
        instrument_type: Type of instrument
        maturity: Maturity date or tenor in years
        market_rate: Quoted market rate
        weight: Weight in calibration (for global fitting)
    """
    instrument_type: InstrumentType
    maturity: Union[date, float]
    market_rate: float
    weight: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BootstrapConfig:
    """Configuration for curve bootstrapping."""
    interpolation: str = "log_linear"
    max_iterations: int = 100
    tolerance: float = 1e-12


@dataclass
class CalibrationResult:
    """Result of curve calibration."""
    curve: Curve
    converged: bool
    iterations: int
    max_error: float
    errors: Dict[str, float] = field(default_factory=dict)


class CurveBootstrapper:
    """
    Curve bootstrapper using sequential stripping.

    Example:
        >>> bootstrapper = CurveBootstrapper(date(2024, 1, 15))
        >>> _ = bootstrapper.add_deposit(0.25, 0.05)
        >>> _ = bootstrapper.add_swap(2.0, 0.045)
        >>> _ = bootstrapper.add_swap(5.0, 0.048)
        >>> result = bootstrapper.solve()
        >>> curve = result.curve
    """

    def __init__(
        self,
        reference_date: date,
        config: Optional[BootstrapConfig] = None,
    ):
        self.reference_date = reference_date
        self.config = config or BootstrapConfig()
        self._instruments: List[CalibrationInstrument] = []

    def add_deposit(
        self,
        maturity: Union[date, float],
        rate: float,
        day_count: str = "ACT/360",
    ) -> "CurveBootstrapper":
        """Add a deposit rate."""
        self._instruments.append(CalibrationInstrument(
            instrument_type=InstrumentType.DEPOSIT,
            maturity=maturity,
            market_rate=rate,
            params={"day_count": day_count},
        ))
        return self

    def add_fra(
        self,
        start: float,
        end: float,
        rate: float,
    ) -> "CurveBootstrapper":
        """Add a FRA."""
        self._instruments.append(CalibrationInstrument(
            instrument_type=InstrumentType.FRA,
            maturity=end,
            market_rate=rate,
            params={"start": start, "end": end},
        ))
        return self

    def add_future(
        self,
        maturity: Union[date, float],
        price: float,
        convexity_adj: float = 0.0,
    ) -> "CurveBootstrapper":
        """Add a futures contract."""
        rate = (100 - price) / 100 - convexity_adj
        self._instruments.append(CalibrationInstrument(
            instrument_type=InstrumentType.FUTURE,
            maturity=maturity,
            market_rate=rate,
            params={"price": price, "convexity_adj": convexity_adj},
        ))
        return self

    def add_swap(
        self,
        maturity: Union[date, float],
        rate: float,
        frequency: int = 2,
        fixed_dc: str = "30/360",
    ) -> "CurveBootstrapper":
        """Add an interest rate swap."""
        self._instruments.append(CalibrationInstrument(
            instrument_type=InstrumentType.SWAP,
            maturity=maturity,
            market_rate=rate,
            params={"frequency": frequency, "fixed_dc": fixed_dc},
        ))
        return self

    def add_ois(
        self,
        maturity: Union[date, float],
        rate: float,
    ) -> "CurveBootstrapper":
        """Add an OIS swap."""
        self._instruments.append(CalibrationInstrument(
            instrument_type=InstrumentType.OIS,
            maturity=maturity,
            market_rate=rate,
        ))
        return self

    def _to_years(self, maturity: Union[date, float]) -> float:
        """Convert maturity to year fraction."""
        if isinstance(maturity, date):
            return (maturity - self.reference_date).days / 365.0
        return maturity

    def _interp_df(
        self,
        tenors: List[float],
        dfs: List[float],
        t: float,
    ) -> float:
        """Log-linear interpolation of discount factor."""
        if t <= tenors[0]:
            return dfs[0]
        if t >= tenors[-1]:
            # Flat extrapolation in log space
            if len(tenors) >= 2:
                slope = math.log(dfs[-1] / dfs[-2]) / (tenors[-1] - tenors[-2])
                return dfs[-1] * math.exp(slope * (t - tenors[-1]))
            return dfs[-1]

        # Find interval
        for i in range(len(tenors) - 1):
            if tenors[i] <= t <= tenors[i + 1]:
                w = (t - tenors[i]) / (tenors[i + 1] - tenors[i])
                log_df = (1 - w) * math.log(dfs[i]) + w * math.log(dfs[i + 1])
                return math.exp(log_df)

        return dfs[-1]

    def _bootstrap_deposit(self, rate: float, t: float) -> float:
        """Bootstrap DF from deposit: DF = 1 / (1 + r*t)"""
        return 1.0 / (1.0 + rate * t)

    def _bootstrap_fra(
        self,
        rate: float,
        start: float,
        end: float,
        tenors: List[float],
        dfs: List[float],
    ) -> float:
        """Bootstrap DF from FRA."""
        df_start = self._interp_df(tenors, dfs, start)
        return df_start / (1.0 + rate * (end - start))

    def _bootstrap_swap(
        self,
        rate: float,
        maturity: float,
        frequency: int,
        tenors: List[float],
        dfs: List[float],
    ) -> float:
        """Bootstrap DF from swap using root finding."""
        period = 1.0 / frequency
        n_periods = int(maturity * frequency)

        # Sum of known DFs
        annuity_known = 0.0
        for i in range(1, n_periods):
            t_i = i * period
            df_i = self._interp_df(tenors, dfs, t_i)
            annuity_known += df_i * period

        # Solve: 1 - rate * (annuity_known + DF_T * period) - DF_T = 0
        # DF_T * (1 + rate * period) = 1 - rate * annuity_known
        df_T = (1.0 - rate * annuity_known) / (1.0 + rate * period)
        return df_T

    def _bootstrap_ois(self, rate: float, t: float) -> float:
        """Bootstrap DF from OIS (simple case)."""
        return 1.0 / (1.0 + rate * t)

    def solve(self) -> CalibrationResult:
        """
        Bootstrap curve from instruments.

        Returns:
            CalibrationResult with bootstrapped curve
        """
        # Sort by maturity
        sorted_insts = sorted(
            self._instruments,
            key=lambda x: self._to_years(x.maturity)
        )

        tenors = [0.0]
        dfs = [1.0]
        errors: Dict[str, float] = {}

        for i, inst in enumerate(sorted_insts):
            t = self._to_years(inst.maturity)
            tenors.append(t)

            if inst.instrument_type == InstrumentType.DEPOSIT:
                df = self._bootstrap_deposit(inst.market_rate, t)

            elif inst.instrument_type == InstrumentType.FRA:
                start = inst.params["start"]
                end = inst.params["end"]
                df = self._bootstrap_fra(inst.market_rate, start, end, tenors[:-1], dfs)

            elif inst.instrument_type == InstrumentType.SWAP:
                freq = inst.params.get("frequency", 2)
                df = self._bootstrap_swap(inst.market_rate, t, freq, tenors[:-1], dfs)

            elif inst.instrument_type == InstrumentType.OIS:
                df = self._bootstrap_ois(inst.market_rate, t)

            elif inst.instrument_type == InstrumentType.FUTURE:
                df = self._bootstrap_deposit(inst.market_rate, t)

            else:
                df = self._bootstrap_deposit(inst.market_rate, t)

            dfs.append(df)
            errors[f"{inst.instrument_type.value}_{t:.2f}"] = 0.0  # Perfect fit by construction

        curve = DiscountCurve(
            reference_date=self.reference_date,
            tenors=tenors,
            values=dfs,
            interpolation=self.config.interpolation,
        )

        return CalibrationResult(
            curve=curve,
            converged=True,
            iterations=1,
            max_error=0.0,
            errors=errors,
        )


class GlobalCurveFitter:
    """
    Global curve fitting using optimization.

    Fits a smooth curve to all instruments simultaneously.
    Better for noisy data or when smoothness is important.

    Example:
        >>> fitter = GlobalCurveFitter(date(2024, 1, 15))
        >>> _ = fitter.add_swap(2.0, 0.045)
        >>> _ = fitter.add_swap(5.0, 0.048)
        >>> result = fitter.solve(n_knots=10)
    """

    def __init__(
        self,
        reference_date: date,
        config: Optional[BootstrapConfig] = None,
    ):
        self.reference_date = reference_date
        self.config = config or BootstrapConfig()
        self._instruments: List[CalibrationInstrument] = []

    def add_deposit(self, maturity: float, rate: float) -> "GlobalCurveFitter":
        self._instruments.append(CalibrationInstrument(
            InstrumentType.DEPOSIT, maturity, rate
        ))
        return self

    def add_swap(
        self,
        maturity: float,
        rate: float,
        frequency: int = 2,
    ) -> "GlobalCurveFitter":
        self._instruments.append(CalibrationInstrument(
            InstrumentType.SWAP, maturity, rate, params={"frequency": frequency}
        ))
        return self

    def _to_years(self, maturity: Union[date, float]) -> float:
        if isinstance(maturity, date):
            return (maturity - self.reference_date).days / 365.0
        return maturity

    def solve(self, n_knots: int = 10) -> CalibrationResult:
        """
        Fit curve globally using least squares.

        Args:
            n_knots: Number of spline knots

        Returns:
            CalibrationResult
        """
        # Get maturity range
        max_mat = max(self._to_years(inst.maturity) for inst in self._instruments)
        if n_knots < 2:
            n_knots = 2
        knot_times = [max_mat * i / (n_knots - 1) for i in range(n_knots)]

        # Initial guess: flat zero rate
        initial_zeros = [0.03] * n_knots

        def residuals(zeros: List[float]) -> List[float]:
            # Build curve from zeros (zeros[0] corresponds to t=0, always DF=1)
            dfs = [1.0] + [math.exp(-z * t) for z, t in zip(zeros[1:], knot_times[1:])]
            # Note: zeros[0] is unused since DF(0)=1 always; optimizer may leave it at initial value

            errors = []
            for inst in self._instruments:
                t = self._to_years(inst.maturity)

                if inst.instrument_type == InstrumentType.DEPOSIT:
                    df = self._interp_df(knot_times, dfs, t)
                    implied = (1.0 / df - 1.0) / t
                    errors.append((implied - inst.market_rate) * inst.weight)

                elif inst.instrument_type == InstrumentType.SWAP:
                    freq = inst.params.get("frequency", 2)
                    period = 1.0 / freq
                    n_periods = int(t * freq)

                    annuity = sum(
                        self._interp_df(knot_times, dfs, i * period) * period
                        for i in range(1, n_periods + 1)
                    )
                    df_T = self._interp_df(knot_times, dfs, t)
                    implied = (1.0 - df_T) / annuity if annuity > 0 else 0
                    errors.append((implied - inst.market_rate) * inst.weight)

            return errors

        def _interp_df(self, tenors, dfs, t):
            if t <= tenors[0]:
                return dfs[0]
            if t >= tenors[-1]:
                return dfs[-1]
            for i in range(len(tenors) - 1):
                if tenors[i] <= t <= tenors[i + 1]:
                    w = (t - tenors[i]) / (tenors[i + 1] - tenors[i])
                    log_df = (1 - w) * math.log(dfs[i]) + w * math.log(dfs[i + 1])
                    return math.exp(log_df)
            return dfs[-1]

        # Make _interp_df available in residuals
        self._interp_df = _interp_df.__get__(self)

        # Solve
        optimizer = LevenbergMarquardt(
            tol=self.config.tolerance,
            maxiter=self.config.max_iterations,
        )
        result = optimizer.solve(residuals, initial_zeros)

        # Build final curve
        final_zeros = result.x
        final_dfs = [1.0] + [math.exp(-z * t) for z, t in zip(final_zeros[1:], knot_times[1:])]

        curve = DiscountCurve(
            reference_date=self.reference_date,
            tenors=knot_times,
            values=final_dfs,
            interpolation=self.config.interpolation,
        )

        return CalibrationResult(
            curve=curve,
            converged=result.converged,
            iterations=result.iterations,
            max_error=result.residual_norm,
        )


# Convenience function
def bootstrap_curve(
    reference_date: date,
    instruments: List[tuple],
    interpolation: str = "log_linear",
) -> DiscountCurve:
    """
    Convenience function to bootstrap a curve.

    Args:
        reference_date: Curve reference date
        instruments: List of (type, maturity, rate) tuples
                    Types: "deposit", "swap", "ois", "fra"
        interpolation: Interpolation method

    Returns:
        Bootstrapped DiscountCurve

    Example:
        >>> curve = bootstrap_curve(
        ...     date(2024, 1, 15),
        ...     [("deposit", 0.25, 0.05), ("swap", 2.0, 0.045), ("swap", 5.0, 0.048)]
        ... )
    """
    bootstrapper = CurveBootstrapper(
        reference_date=reference_date,
        config=BootstrapConfig(interpolation=interpolation),
    )

    for item in instruments:
        if len(item) == 3:
            inst_type, maturity, rate = item
        else:
            raise ValueError(f"Expected (type, maturity, rate), got {item}")

        inst_type = inst_type.lower()
        if inst_type == "deposit":
            bootstrapper.add_deposit(maturity, rate)
        elif inst_type == "swap":
            bootstrapper.add_swap(maturity, rate)
        elif inst_type == "ois":
            bootstrapper.add_ois(maturity, rate)
        elif inst_type == "fra":
            # For FRA, maturity is end, need start
            bootstrapper.add_fra(maturity - 0.25, maturity, rate)

    return bootstrapper.solve().curve
