"""
Standard Result Dataclasses

Provides consistent result types across the library:
- MonteCarloResult: Monte Carlo pricing with statistics
- AnalyticalResult: Analytical pricing with Greeks
- CalibrationResult: Model calibration outcomes
- SimulationResult: Path simulation results

These dataclasses ensure consistent API across different models
and facilitate JSON-friendly serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class MonteCarloResult:
    """
    Result of Monte Carlo pricing.

    Provides price estimate with statistical measures.

    Attributes:
        price: Estimated option price
        std_error: Standard error of the estimate
        n_paths: Number of simulation paths
        confidence_interval: Optional (lower, upper) 95% CI bounds
        convergence_rate: Optional estimated convergence rate
        variance: Sample variance of payoffs
        elapsed_time: Computation time in seconds (optional)
    """
    price: float
    std_error: float
    n_paths: int
    confidence_interval: Optional[tuple[float, float]] = None
    convergence_rate: Optional[float] = None
    variance: Optional[float] = None
    elapsed_time: Optional[float] = None

    def __post_init__(self):
        # Compute 95% CI if not provided
        if self.confidence_interval is None and self.std_error > 0:
            z = 1.96  # 95% confidence
            self.confidence_interval = (
                self.price - z * self.std_error,
                self.price + z * self.std_error
            )

    @property
    def relative_error(self) -> float:
        """Relative standard error as percentage."""
        if abs(self.price) < 1e-10:
            return float('inf') if self.std_error > 0 else 0.0
        return abs(self.std_error / self.price)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "price": self.price,
            "std_error": self.std_error,
            "n_paths": self.n_paths,
            "confidence_interval": list(self.confidence_interval) if self.confidence_interval else None,
            "convergence_rate": self.convergence_rate,
            "variance": self.variance,
            "elapsed_time": self.elapsed_time,
            "relative_error": self.relative_error,
        }


@dataclass
class AnalyticalResult:
    """
    Result of analytical (closed-form) pricing.

    Provides price with optional Greeks.

    Attributes:
        price: Option price
        delta: First derivative w.r.t. spot
        gamma: Second derivative w.r.t. spot
        theta: Time decay (per day)
        vega: Sensitivity to volatility (per 1% change)
        rho: Sensitivity to interest rate (per 1% change)
        model_name: Name of pricing model used
    """
    price: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    model_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "price": self.price,
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "rho": self.rho,
            "model_name": self.model_name,
        }


@dataclass
class FFTPricingResult:
    """
    Result of FFT-based option pricing.

    Attributes:
        price: Option price
        strikes: Array of strikes computed
        prices: Array of prices for all strikes
        n_fft: FFT size used
        alpha: Dampening parameter
        model_name: Name of model used
    """
    price: float
    strikes: Optional[List[float]] = None
    prices: Optional[List[float]] = None
    n_fft: Optional[int] = None
    alpha: Optional[float] = None
    model_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "price": self.price,
            "strikes": self.strikes,
            "prices": self.prices,
            "n_fft": self.n_fft,
            "alpha": self.alpha,
            "model_name": self.model_name,
        }


@dataclass
class SimulationResult:
    """
    Result of path simulation.

    Attributes:
        times: Time grid [t_0, t_1, ..., t_N]
        paths: Simulated paths [path_0, path_1, ...], each path is [S_0, S_1, ..., S_N]
        n_paths: Number of paths
        n_steps: Number of time steps
        dt: Time step size
        terminal_values: Final values S_N for each path
        statistics: Optional dict with mean, std, percentiles at terminal time
    """
    times: List[float]
    paths: List[List[float]]
    n_paths: int
    n_steps: int
    dt: float
    terminal_values: Optional[List[float]] = None
    statistics: Optional[Dict[str, float]] = None

    def __post_init__(self):
        # Compute terminal values if not provided
        if self.terminal_values is None and self.paths:
            self.terminal_values = [path[-1] for path in self.paths]

    def compute_statistics(self) -> Dict[str, float]:
        """Compute summary statistics of terminal distribution."""
        if not self.terminal_values:
            return {}

        values = sorted(self.terminal_values)
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0

        self.statistics = {
            "mean": mean,
            "std": variance ** 0.5,
            "min": values[0],
            "max": values[-1],
            "median": values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2,
            "p05": values[int(0.05 * n)],
            "p25": values[int(0.25 * n)],
            "p75": values[int(0.75 * n)],
            "p95": values[int(0.95 * n)],
        }
        return self.statistics

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary (excluding full paths for size)."""
        if self.statistics is None:
            self.compute_statistics()

        return {
            "n_paths": self.n_paths,
            "n_steps": self.n_steps,
            "dt": self.dt,
            "T": self.times[-1] if self.times else 0,
            "statistics": self.statistics,
        }


@dataclass
class ModelCalibrationResult:
    """
    Result of model calibration.

    Attributes:
        success: Whether calibration converged
        parameters: Calibrated parameter values
        objective_value: Final objective function value
        n_iterations: Number of iterations
        residuals: Fitting errors for each instrument
        message: Status message
    """
    success: bool
    parameters: Dict[str, float]
    objective_value: float
    n_iterations: int
    residuals: Optional[List[float]] = None
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "success": self.success,
            "parameters": self.parameters,
            "objective_value": self.objective_value,
            "n_iterations": self.n_iterations,
            "residuals": self.residuals,
            "message": self.message,
        }


@dataclass
class ImpliedVolResult:
    """
    Result of implied volatility calculation.

    Attributes:
        implied_vol: The implied volatility
        price: Market/model price used
        n_iterations: Newton-Raphson iterations
        converged: Whether the solver converged
    """
    implied_vol: float
    price: float
    n_iterations: int = 0
    converged: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "implied_vol": self.implied_vol,
            "price": self.price,
            "n_iterations": self.n_iterations,
            "converged": self.converged,
        }
