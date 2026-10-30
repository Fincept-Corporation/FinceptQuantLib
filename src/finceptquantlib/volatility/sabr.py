"""
SABR Model Implementation

The SABR (Stochastic Alpha Beta Rho) model for volatility smiles.

dF = σ F^β dW_1
dσ = ν σ dW_2
dW_1 dW_2 = ρ dt

Parameters:
- α (alpha): Initial volatility
- β (beta): CEV exponent (0 = normal, 1 = lognormal)
- ρ (rho): Correlation between forward and vol processes
- ν (nu): Volatility of volatility
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm


@dataclass
class SABRParameters:
    """SABR model parameters."""
    alpha: float  # Initial volatility level
    beta: float   # CEV exponent [0, 1]
    rho: float    # Correlation [-1, 1]
    nu: float     # Vol of vol (volvol)

    def __post_init__(self):
        if not 0 <= self.beta <= 1:
            raise ValueError(f"Beta must be in [0, 1], got {self.beta}")
        if not -1 <= self.rho <= 1:
            raise ValueError(f"Rho must be in [-1, 1], got {self.rho}")
        if self.nu < 0:
            raise ValueError(f"Nu must be non-negative, got {self.nu}")
        if self.alpha <= 0:
            raise ValueError(f"Alpha must be positive, got {self.alpha}")


def sabr_implied_vol(
    forward: float,
    strike: float,
    expiry: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """
    Calculate SABR implied Black volatility.

    Uses Hagan's approximation formula (2002).

    Args:
        forward: Forward rate
        strike: Strike rate
        expiry: Time to expiry
        alpha: Initial vol
        beta: CEV exponent
        rho: Correlation
        nu: Vol of vol

    Returns:
        Implied Black volatility
    """
    # Handle ATM case
    if abs(forward - strike) < 1e-10:
        return _sabr_atm_vol(forward, expiry, alpha, beta, rho, nu)

    # Hagan's formula
    FK = forward * strike
    log_FK = math.log(forward / strike)

    # Avoid numerical issues
    if FK <= 0:
        return alpha

    FK_beta = FK ** ((1 - beta) / 2)

    # z and x(z) terms
    z = (nu / alpha) * FK_beta * log_FK
    if abs(z) < 1e-10:
        x_z = 1.0
    else:
        sqrt_term = math.sqrt(1 - 2 * rho * z + z ** 2)
        # Guard against rho near ±1
        denom_arg = (sqrt_term + z - rho) / (1 - rho) if abs(1 - rho) > 1e-12 else None
        if denom_arg is None or denom_arg <= 0:
            # Limiting case: rho → 1, x(z) → z / (z) = 1 for small correlation correction
            # Use Taylor expansion: x(z) ≈ 1 - rho*z/2 + ...
            x_z = 1.0
        else:
            x_z = z / math.log(denom_arg)

    # Series expansion terms
    one_minus_beta = 1 - beta
    one_minus_beta_sq = one_minus_beta ** 2

    A = alpha / (FK_beta * (1 + one_minus_beta_sq / 24 * log_FK ** 2 +
                           one_minus_beta_sq ** 2 / 1920 * log_FK ** 4))

    B1 = one_minus_beta_sq / 24 * alpha ** 2 / FK_beta ** 2
    B2 = rho * beta * nu * alpha / (4 * FK_beta)
    B3 = (2 - 3 * rho ** 2) * nu ** 2 / 24

    B = 1 + (B1 + B2 + B3) * expiry

    return A * x_z * B


def _sabr_atm_vol(
    forward: float,
    expiry: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """ATM SABR volatility."""
    F_beta = forward ** (1 - beta)

    term1 = (1 - beta) ** 2 / 24 * alpha ** 2 / F_beta ** 2
    term2 = rho * beta * nu * alpha / (4 * F_beta)
    term3 = (2 - 3 * rho ** 2) * nu ** 2 / 24

    return alpha / F_beta * (1 + (term1 + term2 + term3) * expiry)


def sabr_normal_vol(
    forward: float,
    strike: float,
    expiry: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """
    Calculate SABR implied normal (Bachelier) volatility.

    Args:
        forward: Forward rate
        strike: Strike rate
        expiry: Time to expiry
        alpha: Initial vol
        beta: CEV exponent
        rho: Correlation
        nu: Vol of vol

    Returns:
        Implied normal volatility
    """
    # Use Black vol and convert
    black_vol = sabr_implied_vol(forward, strike, expiry, alpha, beta, rho, nu)

    # Approximate conversion: σ_N ≈ σ_B * F for ATM
    avg_FK = (forward + strike) / 2
    return black_vol * avg_FK


class SABRModel:
    """
    SABR volatility model for a single expiry.

    Provides methods for:
    - Implied volatility calculation
    - Greeks (delta, vega, etc.) under SABR
    - Smile dynamics
    """

    def __init__(
        self,
        params: SABRParameters,
        forward: float,
        expiry: float,
    ):
        self.params = params
        self.forward = forward
        self.expiry = expiry

    def implied_vol(self, strike: float) -> float:
        """Get implied Black volatility at strike."""
        return sabr_implied_vol(
            self.forward,
            strike,
            self.expiry,
            self.params.alpha,
            self.params.beta,
            self.params.rho,
            self.params.nu,
        )

    def smile(self, strikes: list[float]) -> list[float]:
        """Get volatility smile across strikes."""
        return [self.implied_vol(k) for k in strikes]

    def atm_vol(self) -> float:
        """Get ATM volatility."""
        return self.implied_vol(self.forward)

    def skew(self, delta_strike: float = 0.01) -> float:
        """
        Calculate volatility skew (∂σ/∂K at ATM).
        """
        vol_up = self.implied_vol(self.forward + delta_strike)
        vol_down = self.implied_vol(self.forward - delta_strike)
        return (vol_up - vol_down) / (2 * delta_strike)

    def curvature(self, delta_strike: float = 0.01) -> float:
        """
        Calculate volatility curvature (∂²σ/∂K² at ATM).
        """
        vol_up = self.implied_vol(self.forward + delta_strike)
        vol_atm = self.implied_vol(self.forward)
        vol_down = self.implied_vol(self.forward - delta_strike)
        return (vol_up - 2 * vol_atm + vol_down) / (delta_strike ** 2)

    def delta_vol(self, bump: float = 0.0001) -> SABRParameters:
        """
        Calculate sensitivity of model to forward move.

        Returns new parameters after forward shift.
        """
        # Under SABR, alpha shifts with forward (backbone)
        new_forward = self.forward + bump
        # Backbone: α' = α * (F'/F)^(1-β)
        alpha_shift = self.params.alpha * (new_forward / self.forward) ** (1 - self.params.beta)

        return SABRParameters(
            alpha=alpha_shift,
            beta=self.params.beta,
            rho=self.params.rho,
            nu=self.params.nu,
        )


def calibrate_sabr(
    forward: float,
    expiry: float,
    strikes: list[float],
    market_vols: list[float],
    beta: Optional[float] = None,
    weights: Optional[list[float]] = None,
    method: str = "least_squares",
) -> SABRParameters:
    """
    Calibrate SABR parameters to market volatilities.

    Args:
        forward: Forward rate
        expiry: Time to expiry
        strikes: Array of strikes
        market_vols: Array of market implied vols
        beta: If provided, fix beta and calibrate alpha, rho, nu
        weights: Optional weights for each strike
        method: Calibration method ("least_squares", "differential_evolution")

    Returns:
        Calibrated SABRParameters
    """
    if weights is None:
        weights = [1.0] * len(strikes)

    # Objective function
    def objective(params):
        if beta is not None:
            alpha, rho, nu = params
            b = beta
        else:
            alpha, b, rho, nu = params

        # Constraints
        if alpha <= 0 or nu < 0 or b < 0 or b > 1 or rho < -1 or rho > 1:
            return 1e10

        model_vols = [
            sabr_implied_vol(forward, k, expiry, alpha, b, rho, nu)
            for k in strikes
        ]

        errors = sum((mv - mkt) ** 2 * w for mv, mkt, w in zip(model_vols, market_vols, weights))
        return errors

    # Initial guess - linear interpolation for ATM vol
    def linear_interp(x, xs, ys):
        for i in range(len(xs) - 1):
            if xs[i] <= x <= xs[i + 1]:
                w = (x - xs[i]) / (xs[i + 1] - xs[i]) if xs[i + 1] != xs[i] else 0
                return ys[i] * (1 - w) + ys[i + 1] * w
        return ys[0] if x < xs[0] else ys[-1]

    atm_vol = linear_interp(forward, list(strikes), list(market_vols))
    if beta is not None:
        x0 = [atm_vol * forward ** (1 - beta), 0.0, 0.3]
        bounds = [(1e-6, 2.0), (-0.99, 0.99), (1e-6, 2.0)]
    else:
        x0 = [atm_vol * forward ** 0.5, 0.5, 0.0, 0.3]
        bounds = [(1e-6, 2.0), (0.01, 0.99), (-0.99, 0.99), (1e-6, 2.0)]

    # Optimize
    if method == "differential_evolution":
        result = differential_evolution(objective, bounds, seed=42)
    else:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

    # Extract parameters
    if beta is not None:
        alpha, rho, nu = result.x
        b = beta
    else:
        alpha, b, rho, nu = result.x

    return SABRParameters(alpha=alpha, beta=b, rho=rho, nu=nu)


def sabr_density(
    forward: float,
    expiry: float,
    params: SABRParameters,
    strikes: list[float],
) -> list[float]:
    """
    Calculate risk-neutral density implied by SABR.

    Uses Breeden-Litzenberger formula:
    p(K) = e^(rT) * d²C/dK²
    """
    # Get call prices from Black with SABR vol
    def call_price(k):
        vol = sabr_implied_vol(
            forward, k, expiry,
            params.alpha, params.beta, params.rho, params.nu
        )
        d1 = (math.log(forward / k) + 0.5 * vol ** 2 * expiry) / (vol * math.sqrt(expiry))
        d2 = d1 - vol * math.sqrt(expiry)
        return forward * norm.cdf(d1) - k * norm.cdf(d2)

    # Numerical second derivative
    dk = 0.001 * forward
    density = [0.0] * len(strikes)

    for i, k in enumerate(strikes):
        c_up = call_price(k + dk)
        c_mid = call_price(k)
        c_down = call_price(k - dk)
        density[i] = (c_up - 2 * c_mid + c_down) / (dk ** 2)

    # Normalize using trapezoidal rule
    def trapz(y, x):
        total = 0.0
        for i in range(len(x) - 1):
            total += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
        return total

    total = trapz(density, strikes)
    if total > 0:
        density = [d / total for d in density]

    return density


class SABRSmileDynamics:
    """
    Analyze SABR smile dynamics.

    Studies how the smile evolves with:
    - Forward moves (sticky strike vs sticky delta)
    - Time decay
    - Vol of vol changes
    """

    def __init__(self, params: SABRParameters, forward: float, expiry: float):
        self.params = params
        self.forward = forward
        self.expiry = expiry
        self._model = SABRModel(params, forward, expiry)

    def sticky_strike_vol(
        self,
        strike: float,
        new_forward: float,
    ) -> float:
        """
        Calculate vol at strike after forward move (sticky strike).

        Strike stays fixed, forward moves.
        """
        # Adjust alpha for backbone
        new_alpha = self.params.alpha * (new_forward / self.forward) ** (1 - self.params.beta)

        return sabr_implied_vol(
            new_forward, strike, self.expiry,
            new_alpha, self.params.beta, self.params.rho, self.params.nu
        )

    def sticky_delta_vol(
        self,
        delta: float,
        new_forward: float,
    ) -> float:
        """
        Calculate vol at delta after forward move (sticky delta).

        Strike moves to maintain same delta.
        """
        # Find the strike with the same delta at the new forward
        new_strike = self._strike_from_delta(delta, new_forward)

        # Adjust alpha
        new_alpha = self.params.alpha * (new_forward / self.forward) ** (1 - self.params.beta)

        return sabr_implied_vol(
            new_forward, new_strike, self.expiry,
            new_alpha, self.params.beta, self.params.rho, self.params.nu
        )

    def _strike_from_delta(self, delta: float, forward: float) -> float:
        """Find strike for given Black delta."""
        from scipy.optimize import brentq

        def delta_error(k):
            vol = sabr_implied_vol(
                forward, k, self.expiry,
                self.params.alpha, self.params.beta, self.params.rho, self.params.nu
            )
            d1 = (math.log(forward / k) + 0.5 * vol ** 2 * self.expiry) / (vol * math.sqrt(self.expiry))
            return norm.cdf(d1) - delta

        # Search bounds
        low = forward * 0.5
        high = forward * 1.5

        try:
            return brentq(delta_error, low, high)
        except ValueError:
            return forward  # Fallback to ATM

    def time_decay(self, new_expiry: float) -> SABRModel:
        """Get model with shorter expiry (time decay)."""
        return SABRModel(self.params, self.forward, new_expiry)
