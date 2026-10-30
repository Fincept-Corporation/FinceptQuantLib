"""
Ito Calculus Tools

Theoretical tools for stochastic calculus:
- Ito's lemma application
- Quadratic variation
- Stochastic integrals (numerical approximation)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Tuple


@dataclass
class ItoProcess:
    """
    Representation of an Ito process.

    dX = μ(X,t)dt + σ(X,t)dW

    Stores the drift μ and diffusion σ functions.
    """
    drift: Callable[[float, float], float]  # μ(x, t)
    diffusion: Callable[[float, float], float]  # σ(x, t)
    X0: float = 0.0

    def __repr__(self) -> str:
        return f"ItoProcess(X0={self.X0})"


def ito_lemma_1d(
    f: Callable[[float, float], float],
    f_t: Callable[[float, float], float],
    f_x: Callable[[float, float], float],
    f_xx: Callable[[float, float], float],
    mu: Callable[[float, float], float],
    sigma: Callable[[float, float], float]
) -> Tuple[Callable[[float, float], float], Callable[[float, float], float]]:
    """
    Apply Ito's lemma to function f(X,t) where dX = μdt + σdW.

    Ito's lemma:
    df = (f_t + μf_x + ½σ²f_xx)dt + σf_x dW

    So Y = f(X,t) satisfies:
    dY = μ_Y dt + σ_Y dW

    where:
    μ_Y = f_t + μf_x + ½σ²f_xx
    σ_Y = σf_x

    Args:
        f: Function f(x, t)
        f_t: ∂f/∂t
        f_x: ∂f/∂x
        f_xx: ∂²f/∂x²
        mu: Original drift μ(x, t)
        sigma: Original diffusion σ(x, t)

    Returns:
        (mu_Y, sigma_Y): Drift and diffusion of Y = f(X,t)

    Example:
        >>> # For GBM with f(S,t) = log(S)
        >>> # f_t = 0, f_x = 1/S, f_xx = -1/S²
        >>> # dlog(S) = (μ - σ²/2)dt + σdW
        >>> f = lambda s, t: math.log(s)
        >>> f_t = lambda s, t: 0
        >>> f_x = lambda s, t: 1/s
        >>> f_xx = lambda s, t: -1/s**2
        >>> mu = lambda s, t: 0.05 * s
        >>> sigma = lambda s, t: 0.2 * s
        >>> mu_log, sigma_log = ito_lemma_1d(f, f_t, f_x, f_xx, mu, sigma)
    """
    def mu_Y(x: float, t: float) -> float:
        s = sigma(x, t)
        return f_t(x, t) + mu(x, t) * f_x(x, t) + 0.5 * s ** 2 * f_xx(x, t)

    def sigma_Y(x: float, t: float) -> float:
        return sigma(x, t) * f_x(x, t)

    return mu_Y, sigma_Y


def ito_product_rule(
    mu_X: Callable[[float, float, float], float],
    sigma_X: Callable[[float, float, float], float],
    mu_Y: Callable[[float, float, float], float],
    sigma_Y: Callable[[float, float, float], float],
    rho: float = 0.0
) -> Tuple[Callable, Callable]:
    """
    Ito product rule: d(XY) = X dY + Y dX + d<X,Y>

    where d<X,Y> = ρ σ_X σ_Y dt is the covariation.

    For dX = μ_X dt + σ_X dW_1
        dY = μ_Y dt + σ_Y dW_2
        Corr(dW_1, dW_2) = ρ

    d(XY) = (Xμ_Y + Yμ_X + ρσ_Xσ_Y)dt + Xσ_Y dW_2 + Yσ_X dW_1

    Args:
        mu_X, sigma_X: Drift and diffusion of X(x, y, t)
        mu_Y, sigma_Y: Drift and diffusion of Y(x, y, t)
        rho: Correlation between driving Brownians

    Returns:
        (mu_XY, sigma_XY): Drift of XY (diffusion is more complex)
    """
    def mu_XY(x: float, y: float, t: float) -> float:
        return (x * mu_Y(x, y, t) + y * mu_X(x, y, t) +
                rho * sigma_X(x, y, t) * sigma_Y(x, y, t))

    # Note: The diffusion of XY is a 2D vector, simplified here
    def sigma_XY(x: float, y: float, t: float) -> float:
        # Just the magnitude for scalar approximation
        s1 = x * sigma_Y(x, y, t)
        s2 = y * sigma_X(x, y, t)
        return math.sqrt(s1 ** 2 + s2 ** 2 + 2 * rho * s1 * s2)

    return mu_XY, sigma_XY


def quadratic_variation(
    path: List[float],
    times: List[float]
) -> float:
    """
    Compute empirical quadratic variation of a path.

    [X]_T = lim Σ(X_{t_i} - X_{t_{i-1}})²

    For Brownian motion: [W]_T = T
    For Ito integral: [∫σdW]_T = ∫σ²dt

    Args:
        path: Discrete path values
        times: Corresponding time points

    Returns:
        Quadratic variation [X]_T
    """
    qv = 0.0
    for i in range(1, len(path)):
        qv += (path[i] - path[i - 1]) ** 2
    return qv


def covariation(
    path_X: List[float],
    path_Y: List[float],
    times: List[float]
) -> float:
    """
    Compute empirical covariation (cross-variation) of two paths.

    <X, Y>_T = lim Σ(X_{t_i} - X_{t_{i-1}})(Y_{t_i} - Y_{t_{i-1}})

    Polarization identity: <X,Y> = ½([X+Y] - [X] - [Y])

    Args:
        path_X, path_Y: Discrete path values
        times: Time points

    Returns:
        Covariation <X, Y>_T
    """
    cov = 0.0
    for i in range(1, len(path_X)):
        dX = path_X[i] - path_X[i - 1]
        dY = path_Y[i] - path_Y[i - 1]
        cov += dX * dY
    return cov


def stochastic_integral_approx(
    integrand: Callable[[float, float], float],
    path: List[float],
    times: List[float],
    method: str = "ito"
) -> float:
    """
    Approximate stochastic integral ∫f(X_t, t) dW_t.

    Using sum: Σ f(X_{t_i}, t_i) ΔW_i

    Since we don't have ΔW directly, we approximate
    ΔW ≈ (X_{i+1} - X_i - μΔt) / σ for simple diffusions.

    For general use, this computes ∫f(X,t) dX instead.

    Args:
        integrand: f(x, t) function
        path: Process values
        times: Time points
        method: "ito" (left point), "stratonovich" (midpoint)

    Returns:
        Approximate integral value
    """
    integral = 0.0
    n = len(path) - 1

    for i in range(n):
        dX = path[i + 1] - path[i]

        if method == "ito":
            # Ito integral: use left point
            f = integrand(path[i], times[i])
        elif method == "stratonovich":
            # Stratonovich: use midpoint
            x_mid = 0.5 * (path[i] + path[i + 1])
            t_mid = 0.5 * (times[i] + times[i + 1])
            f = integrand(x_mid, t_mid)
        else:
            raise ValueError(f"Unknown method: {method}")

        integral += f * dX

    return integral


def ito_to_stratonovich(
    mu_ito: Callable[[float, float], float],
    sigma: Callable[[float, float], float],
    sigma_x: Callable[[float, float], float]
) -> Callable[[float, float], float]:
    """
    Convert Ito drift to Stratonovich drift.

    Ito: dX = μ_I dt + σ dW
    Stratonovich: dX = μ_S dt + σ ∘ dW

    μ_S = μ_I - ½ σ σ_x

    Args:
        mu_ito: Ito drift μ_I(x, t)
        sigma: Diffusion σ(x, t)
        sigma_x: ∂σ/∂x

    Returns:
        Stratonovich drift μ_S(x, t)
    """
    def mu_strat(x: float, t: float) -> float:
        return mu_ito(x, t) - 0.5 * sigma(x, t) * sigma_x(x, t)

    return mu_strat


def stratonovich_to_ito(
    mu_strat: Callable[[float, float], float],
    sigma: Callable[[float, float], float],
    sigma_x: Callable[[float, float], float]
) -> Callable[[float, float], float]:
    """
    Convert Stratonovich drift to Ito drift.

    μ_I = μ_S + ½ σ σ_x
    """
    def mu_ito(x: float, t: float) -> float:
        return mu_strat(x, t) + 0.5 * sigma(x, t) * sigma_x(x, t)

    return mu_ito
