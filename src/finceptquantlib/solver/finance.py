"""
Finance-Specific Solvers

One-liner convenience functions for common finance calculations:
- irr: Internal rate of return
- bond_yield: Yield to maturity
- implied_vol: Black-Scholes implied volatility
- par_rate: Par swap rate from curve
- forward_rate: Forward rate from curve
- z_spread: Z-spread over curve
- oas: Option-adjusted spread
- asw_spread: Asset swap spread
- convexity_adjustment: Futures vs forward adjustment
- calibrate_hull_white: Hull-White model calibration
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal, Callable, Union
import math

from finceptquantlib.solver.root1d import brent, newton
from finceptquantlib.solver.leastsq import LevenbergMarquardt


def irr(
    cashflows: List[float],
    times: Optional[List[float]] = None,
    guess: float = 0.1,
    tol: float = 1e-10,
) -> float:
    """
    Compute internal rate of return.

    Args:
        cashflows: Cash flows (first typically negative)
        times: Times of cash flows (defaults to 0, 1, 2, ...)
        guess: Initial guess for rate
        tol: Convergence tolerance

    Returns:
        IRR as decimal (e.g., 0.08 for 8%)

    Raises:
        ValueError: If no IRR exists or doesn't converge
    """
    if times is None:
        times = list(range(len(cashflows)))

    def npv(r: float) -> float:
        # (1 + r) ** t is complex for fractional t once r <= -100%, so a Newton
        # step out of the IRR domain must stop rather than return a complex root.
        if r <= -1:
            raise ValueError("Cannot find IRR - the rate search fell to -100% or below")
        return sum(cf / (1 + r) ** t for cf, t in zip(cashflows, times))

    # Try to find bracket
    r_low, r_high = -0.99, 10.0
    npv_low, npv_high = npv(r_low), npv(r_high)

    # Narrow bracket
    if npv_low * npv_high > 0:
        # Try Newton from guess
        result = newton(npv, guess, tol=tol)
        if result.converged:
            return result.root
        raise ValueError("Cannot find IRR - NPV doesn't cross zero")

    result = brent(npv, r_low, r_high, tol=tol)
    if result.converged:
        return result.root
    raise ValueError(f"IRR did not converge: {result.function_value}")


def xirr(
    cashflows: List[float],
    dates: List[float],
    guess: float = 0.1,
    tol: float = 1e-10,
) -> float:
    """
    Compute IRR with irregular dates (XIRR).

    Args:
        cashflows: Cash flows
        dates: Dates as year fractions from first date
        guess: Initial guess for rate
        tol: Convergence tolerance

    Returns:
        XIRR as decimal
    """
    return irr(cashflows, dates, guess, tol)


def bond_yield(
    price: float,
    face: float,
    coupon: float,
    maturity: float,
    frequency: int = 2,
    guess: float = 0.05,
    tol: float = 1e-10,
) -> float:
    """
    Compute yield to maturity for a fixed-rate bond.

    Args:
        price: Clean price
        face: Face value
        coupon: Annual coupon rate (e.g., 0.05 for 5%)
        maturity: Time to maturity in years
        frequency: Coupon frequency per year
        guess: Initial guess for yield
        tol: Convergence tolerance

    Returns:
        Yield to maturity as decimal

    Example:
        >>> bond_yield(price=98.5, face=100, coupon=0.05, maturity=5, frequency=2)
        0.0534...
    """
    n_periods = int(maturity * frequency)
    period_coupon = face * coupon / frequency

    def pv(y: float) -> float:
        """Present value at yield y."""
        period_yield = y / frequency
        if abs(period_yield) < 1e-12:
            return period_coupon * n_periods + face - price

        pv_coupons = period_coupon * (1 - (1 + period_yield) ** (-n_periods)) / period_yield
        pv_face = face / (1 + period_yield) ** n_periods
        return pv_coupons + pv_face - price

    # Bracket search
    y_low, y_high = -0.5, 2.0
    result = brent(pv, y_low, y_high, tol=tol)

    if result.converged:
        return result.root
    raise ValueError(f"Bond yield did not converge: {result.function_value}")


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: Literal["call", "put"] = "call",
    tol: float = 1e-10,
) -> float:
    """
    Compute Black-Scholes implied volatility.

    Delegates to finceptquantlib.pricing.bs_implied_vol (Newton-Raphson with
    bisection fallback).

    Args:
        price: Market option price
        S: Spot price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate
        q: Dividend yield
        option_type: "call" or "put"

    Returns:
        Implied volatility as decimal
    """
    from finceptquantlib.pricing.analytical.black_scholes import bs_implied_vol
    from finceptquantlib.core.foundations.types import OptionType

    opt_type = OptionType(option_type)
    return bs_implied_vol(price, S, K, r, q, T, opt_type, tol=tol)


def implied_vol_black76(
    price: float,
    F: float,
    K: float,
    T: float,
    r: float,
    option_type: Literal["call", "put"] = "call",
    tol: float = 1e-10,
) -> float:
    """
    Compute Black76 implied volatility for futures/forwards.

    Delegates to finceptquantlib.pricing.black76_implied_vol.

    Args:
        price: Market option price
        F: Forward price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate (for discounting)
        option_type: "call" or "put"

    Returns:
        Implied volatility as decimal
    """
    from finceptquantlib.pricing.analytical.black76 import black76_implied_vol
    from finceptquantlib.core.foundations.types import OptionType

    opt_type = OptionType(option_type)
    return black76_implied_vol(price, F, K, r, T, opt_type, tol=tol)


def par_rate(
    discount_factors: List[float],
    times: List[float],
    frequency: int = 2,
) -> float:
    """
    Compute par swap rate from discount factors.

    Args:
        discount_factors: Discount factors at payment times
        times: Payment times in years
        frequency: Payment frequency per year

    Returns:
        Par swap rate
    """
    if len(discount_factors) != len(times):
        raise ValueError("discount_factors and times must have same length")

    period = 1.0 / frequency
    annuity = sum(df * period for df in discount_factors)

    if abs(annuity) < 1e-15:
        raise ValueError("Zero annuity")

    return (1.0 - discount_factors[-1]) / annuity


def forward_rate(
    df1: float,
    df2: float,
    t1: float,
    t2: float,
    compounding: Literal["simple", "continuous"] = "simple",
) -> float:
    """
    Compute forward rate between two times.

    Args:
        df1: Discount factor at t1
        df2: Discount factor at t2
        t1: Start time
        t2: End time
        compounding: "simple" or "continuous"

    Returns:
        Forward rate for period [t1, t2]
    """
    if t2 <= t1:
        raise ValueError("t2 must be greater than t1")

    dt = t2 - t1

    if compounding == "continuous":
        return -math.log(df2 / df1) / dt
    else:  # simple
        return (df1 / df2 - 1.0) / dt


def zero_rate(
    df: float,
    t: float,
    compounding: Literal["simple", "continuous", "annual"] = "continuous",
) -> float:
    """
    Compute zero rate from discount factor.

    Args:
        df: Discount factor
        t: Time in years
        compounding: Rate convention

    Returns:
        Zero rate
    """
    if t <= 0:
        return 0.0

    if compounding == "continuous":
        return -math.log(df) / t
    elif compounding == "simple":
        return (1.0 / df - 1.0) / t
    else:  # annual
        return df ** (-1.0 / t) - 1.0


def discount_factor(
    rate: float,
    t: float,
    compounding: Literal["simple", "continuous", "annual"] = "continuous",
) -> float:
    """
    Compute discount factor from zero rate.

    Args:
        rate: Zero rate
        t: Time in years
        compounding: Rate convention

    Returns:
        Discount factor
    """
    if t <= 0:
        return 1.0

    if compounding == "continuous":
        return math.exp(-rate * t)
    elif compounding == "simple":
        return 1.0 / (1.0 + rate * t)
    else:  # annual
        return (1.0 + rate) ** (-t)


def duration_from_yield(
    face: float,
    coupon: float,
    maturity: float,
    ytm: float,
    frequency: int = 2,
) -> float:
    """
    Compute Macaulay duration from yield.

    Args:
        face: Face value
        coupon: Annual coupon rate
        maturity: Time to maturity in years
        ytm: Yield to maturity
        frequency: Coupon frequency per year

    Returns:
        Macaulay duration in years
    """
    n_periods = int(maturity * frequency)
    period_coupon = face * coupon / frequency
    period_yield = ytm / frequency

    pv_total = 0.0
    duration_weighted = 0.0

    for i in range(1, n_periods + 1):
        t = i / frequency
        cf = period_coupon if i < n_periods else period_coupon + face
        pv = cf / (1 + period_yield) ** i
        pv_total += pv
        duration_weighted += t * pv

    return duration_weighted / pv_total if pv_total > 0 else 0.0


def modified_duration(
    face: float,
    coupon: float,
    maturity: float,
    ytm: float,
    frequency: int = 2,
) -> float:
    """
    Compute modified duration.

    Args:
        face: Face value
        coupon: Annual coupon rate
        maturity: Time to maturity in years
        ytm: Yield to maturity
        frequency: Coupon frequency per year

    Returns:
        Modified duration
    """
    mac_dur = duration_from_yield(face, coupon, maturity, ytm, frequency)
    return mac_dur / (1 + ytm / frequency)


def convexity(
    face: float,
    coupon: float,
    maturity: float,
    ytm: float,
    frequency: int = 2,
) -> float:
    """
    Compute bond convexity.

    Args:
        face: Face value
        coupon: Annual coupon rate
        maturity: Time to maturity in years
        ytm: Yield to maturity
        frequency: Coupon frequency per year

    Returns:
        Convexity
    """
    n_periods = int(maturity * frequency)
    period_coupon = face * coupon / frequency
    period_yield = ytm / frequency
    y_plus_1 = 1 + period_yield

    pv_total = 0.0
    convexity_sum = 0.0

    for i in range(1, n_periods + 1):
        cf = period_coupon if i < n_periods else period_coupon + face
        pv = cf / y_plus_1 ** i
        pv_total += pv
        convexity_sum += i * (i + 1) * pv

    if pv_total <= 0:
        return 0.0

    return convexity_sum / (pv_total * y_plus_1 ** 2 * frequency ** 2)


# =============================================================================
# Spread Solvers
# =============================================================================

def z_spread(
    price: float,
    cashflows: List[float],
    times: List[float],
    discount_curve: Callable[[float], float],
    tol: float = 1e-10,
) -> float:
    """
    Compute Z-spread (zero-volatility spread) over a discount curve.

    Z-spread is the constant spread added to the spot curve that makes
    the present value of bond cashflows equal to its market price.

    Args:
        price: Market price (dirty price)
        cashflows: Bond cash flows
        times: Times of cash flows in years
        discount_curve: Function that returns discount factor for time t
        tol: Convergence tolerance

    Returns:
        Z-spread in decimal (e.g., 0.005 for 50bp)

    Example:
        >>> curve = lambda t: math.exp(-0.05 * t)
        >>> z_spread(98.0, [5, 5, 105], [1, 2, 3], curve)
        0.00585...
    """
    def pv_error(spread: float) -> float:
        pv = sum(
            cf * discount_curve(t) * math.exp(-spread * t)
            for cf, t in zip(cashflows, times)
        )
        return pv - price

    # Bracket: -500bp to +2000bp
    result = brent(pv_error, -0.05, 0.20, tol=tol)

    if result.converged:
        return result.root
    raise ValueError(f"Z-spread did not converge: {result.function_value}")


def oas(
    price: float,
    cashflows: List[List[float]],
    times: List[float],
    probabilities: List[float],
    discount_curve: Callable[[float], float],
    tol: float = 1e-10,
) -> float:
    """
    Compute Option-Adjusted Spread (OAS).

    OAS is the spread that equates the model price (across scenarios)
    to the market price. Used for bonds with embedded options.

    Args:
        price: Market price (dirty price)
        cashflows: List of cashflow scenarios (one list per scenario)
        times: Times of cash flows in years
        probabilities: Probability of each scenario (must sum to 1)
        discount_curve: Function that returns discount factor for time t
        tol: Convergence tolerance

    Returns:
        OAS in decimal

    Example:
        >>> # Bond callable at par after year 2
        >>> cf_no_call = [5, 5, 5, 105]  # Not called
        >>> cf_called = [5, 5, 100, 0]   # Called at year 2
        >>> times = [1, 2, 3, 4]
        >>> probs = [0.7, 0.3]
        >>> curve = lambda t: math.exp(-0.05 * t)
        >>> oas(98.0, [cf_no_call, cf_called], times, probs, curve)
        0.000854
    """
    if abs(sum(probabilities) - 1.0) > 1e-6:
        raise ValueError("Probabilities must sum to 1")

    def pv_error(spread: float) -> float:
        total_pv = 0.0
        for cf_scenario, prob in zip(cashflows, probabilities):
            scenario_pv = sum(
                cf * discount_curve(t) * math.exp(-spread * t)
                for cf, t in zip(cf_scenario, times)
                if cf != 0
            )
            total_pv += prob * scenario_pv
        return total_pv - price

    result = brent(pv_error, -0.05, 0.20, tol=tol)

    if result.converged:
        return result.root
    raise ValueError(f"OAS did not converge: {result.function_value}")


def asw_spread(
    bond_price: float,
    face: float,
    coupon: float,
    maturity: float,
    frequency: int,
    discount_curve: Callable[[float], float],
    float_curve: Optional[Callable[[float], float]] = None,
) -> float:
    """
    Compute the par-par Asset Swap Spread (ASW spread).

    The investor pays par for the bond plus a swap in which they pay the
    bond's coupons and receive the floating rate + spread on the face amount.
    The spread is the one that makes that package worth par:

        price + float_pv + spread * annuity * face - coupons_pv = face

    which is linear in the spread, so it is solved in closed form. With a
    single curve float_pv = face * (1 - df_T) and this reduces to the usual
    (model bond PV - price) / (annuity * face).

    Args:
        bond_price: Market price (dirty price, per 100 face)
        face: Face value (typically 100)
        coupon: Annual coupon rate (e.g., 0.05 for 5%)
        maturity: Time to maturity in years
        frequency: Coupon frequency per year
        discount_curve: Function returning discount factor
        float_curve: Optional separate curve projecting the floating
                    rate (defaults to discount_curve)

    Returns:
        ASW spread in decimal

    Example:
        >>> curve = lambda t: math.exp(-0.05 * t)
        >>> asw_spread(98.0, 100, 0.05, 5, 2, curve)
        0.003948
    """
    if float_curve is None:
        float_curve = discount_curve

    n_periods = int(maturity * frequency)
    if n_periods < 1:
        raise ValueError("maturity * frequency must be at least one coupon period")
    period = 1.0 / frequency
    period_coupon = face * coupon / frequency
    price = bond_price * face / 100.0  # bond_price is quoted per 100 face

    times = [i * period for i in range(1, n_periods + 1)]
    coupons_pv = sum(period_coupon * discount_curve(t) for t in times)
    annuity = sum(discount_curve(t) * period for t in times)
    # Floating leg without spread: forwards projected off float_curve,
    # discounted on discount_curve.
    float_pv = face * sum(
        (float_curve(t - period) / float_curve(t) - 1.0) * discount_curve(t)
        for t in times
    )

    return (face - price - float_pv + coupons_pv) / (annuity * face)


def i_spread(
    bond_yield: float,
    swap_rate: float,
) -> float:
    """
    Compute I-spread (interpolated spread).

    Simple spread of bond yield over swap rate of same maturity.

    Args:
        bond_yield: Bond yield to maturity
        swap_rate: Swap rate of same maturity

    Returns:
        I-spread in decimal
    """
    return bond_yield - swap_rate


def g_spread(
    bond_yield: float,
    govt_yield: float,
) -> float:
    """
    Compute G-spread (government spread).

    Spread of bond yield over government bond yield of same maturity.

    Args:
        bond_yield: Corporate bond yield
        govt_yield: Government bond yield of same maturity

    Returns:
        G-spread in decimal
    """
    return bond_yield - govt_yield


# =============================================================================
# Convexity Adjustment
# =============================================================================

def convexity_adjustment(
    forward_rate: float,
    volatility: float,
    t1: float,
    t2: float,
    model: Literal["simple", "hull_white"] = "simple",
    mean_reversion: float = 0.03,
) -> float:
    """
    Compute convexity adjustment for futures vs forward rates.

    Futures rates are higher than forward rates due to daily margining.
    This computes the adjustment: forward_rate = futures_rate - adjustment

    Args:
        forward_rate: Forward rate (or futures rate as starting point)
        volatility: Rate volatility (annualized)
        t1: Time to futures expiry
        t2: Time to underlying maturity
        model: "simple" or "hull_white"
        mean_reversion: Mean reversion speed (for hull_white model)

    Returns:
        Convexity adjustment in decimal

    Example:
        >>> # 3M futures expiring in 1 year
        >>> convexity_adjustment(0.05, 0.01, 1.0, 1.25)
        6.25e-05
    """
    tau = t2 - t1  # Underlying period
    if tau <= 0:
        raise ValueError("t2 must be greater than t1")

    if model == "simple":
        # Ho-Lee: adjustment = 0.5 * sigma^2 * t1 * t2
        return 0.5 * volatility ** 2 * t1 * t2

    elif model == "hull_white":
        # Hull-White adjustment (Hull, "Options, Futures, and Other Derivatives"):
        # B(t1,t2)/tau * [B(t1,t2)(1 - e^{-2a t1}) + 2a B(0,t1)^2] * sigma^2 / (4a)
        a = mean_reversion
        sigma = volatility

        if abs(a) < 1e-10:
            return 0.5 * sigma ** 2 * t1 * t2

        b_12 = (1 - math.exp(-a * tau)) / a
        b_01 = (1 - math.exp(-a * t1)) / a
        bracket = b_12 * (1 - math.exp(-2 * a * t1)) + 2 * a * b_01 ** 2
        return (b_12 / tau) * bracket * sigma ** 2 / (4 * a)

    else:
        raise ValueError(f"Unknown model: {model}")


def futures_to_forward(
    futures_rate: float,
    volatility: float,
    t1: float,
    t2: float,
    model: Literal["simple", "hull_white"] = "simple",
    mean_reversion: float = 0.03,
) -> float:
    """
    Convert futures rate to forward rate.

    Args:
        futures_rate: Quoted futures rate
        volatility: Rate volatility
        t1: Time to futures expiry
        t2: Time to underlying maturity
        model: Adjustment model
        mean_reversion: Mean reversion (for hull_white)

    Returns:
        Forward rate
    """
    adj = convexity_adjustment(futures_rate, volatility, t1, t2, model, mean_reversion)
    return futures_rate - adj


def forward_to_futures(
    forward_rate: float,
    volatility: float,
    t1: float,
    t2: float,
    model: Literal["simple", "hull_white"] = "simple",
    mean_reversion: float = 0.03,
) -> float:
    """
    Convert forward rate to futures rate.

    Args:
        forward_rate: Forward rate
        volatility: Rate volatility
        t1: Time to futures expiry
        t2: Time to underlying maturity
        model: Adjustment model
        mean_reversion: Mean reversion (for hull_white)

    Returns:
        Futures rate
    """
    adj = convexity_adjustment(forward_rate, volatility, t1, t2, model, mean_reversion)
    return forward_rate + adj


# =============================================================================
# Model Calibration
# =============================================================================

@dataclass
class HullWhiteParams:
    """Hull-White model parameters."""
    mean_reversion: float  # a
    volatility: float      # sigma


@dataclass
class CalibrationResult:
    """Result of model calibration."""
    params: Union[HullWhiteParams, dict]
    converged: bool
    iterations: int
    error: float
    message: str = ""


def calibrate_hull_white(
    swaption_expiries: List[float],
    swaption_tenors: List[float],
    market_vols: List[float],
    discount_curve: Callable[[float], float],
    initial_a: float = 0.03,
    initial_sigma: float = 0.01,
    weights: Optional[List[float]] = None,
) -> CalibrationResult:
    """
    Calibrate Hull-White model to swaption volatilities.

    Hull-White model: dr = (theta(t) - a*r) dt + sigma dW

    Args:
        swaption_expiries: Swaption expiry times
        swaption_tenors: Underlying swap tenors
        market_vols: Market swaption volatilities (normal vols)
        discount_curve: Discount factor function
        initial_a: Initial guess for mean reversion
        initial_sigma: Initial guess for volatility
        weights: Optional weights for each swaption

    Returns:
        CalibrationResult with HullWhiteParams

    Example:
        >>> expiries = [1, 2, 3, 5]
        >>> tenors = [5, 5, 5, 5]
        >>> vols = [0.0050, 0.0055, 0.0058, 0.0060]
        >>> curve = lambda t: math.exp(-0.05 * t)
        >>> result = calibrate_hull_white(expiries, tenors, vols, curve)
    """
    if weights is None:
        weights = [1.0] * len(market_vols)

    def hw_swaption_vol(a: float, sigma: float, T: float, tenor: float) -> float:
        """Approximate Hull-White swaption volatility."""
        if abs(a) < 1e-10:
            # Limit as a -> 0
            return sigma * math.sqrt(T)

        # B(T, T+tenor)
        B = (1 - math.exp(-a * tenor)) / a

        # Variance of swap rate
        var = (sigma ** 2 / (2 * a ** 3)) * (
            2 * a * T
            - 3 + 4 * math.exp(-a * T)
            - math.exp(-2 * a * T)
        ) * B ** 2 / tenor ** 2

        return math.sqrt(max(var, 0) / T) if T > 0 else 0

    def residuals(params: List[float]) -> List[float]:
        a, sigma = params[0], params[1]
        if a <= 0 or sigma <= 0:
            return [1e10] * len(market_vols)

        errors = []
        for exp, tenor, mkt_vol, w in zip(
            swaption_expiries, swaption_tenors, market_vols, weights
        ):
            model_vol = hw_swaption_vol(a, sigma, exp, tenor)
            errors.append(w * (model_vol - mkt_vol))
        return errors

    optimizer = LevenbergMarquardt(tol=1e-10, maxiter=200)
    result = optimizer.solve(residuals, [initial_a, initial_sigma])

    a_opt, sigma_opt = result.x

    return CalibrationResult(
        params=HullWhiteParams(mean_reversion=a_opt, volatility=sigma_opt),
        converged=result.converged,
        iterations=result.iterations,
        error=result.residual_norm,
        message=result.message,
    )


def calibrate_vasicek(
    zero_rates: List[float],
    times: List[float],
    initial_a: float = 0.1,
    initial_b: float = 0.05,
    initial_sigma: float = 0.01,
) -> CalibrationResult:
    """
    Calibrate Vasicek model to zero rates.

    Vasicek model: dr = a(b - r) dt + sigma dW

    Args:
        zero_rates: Observed zero rates
        times: Times for zero rates
        initial_a: Initial mean reversion speed
        initial_b: Initial long-term mean
        initial_sigma: Initial volatility

    Returns:
        CalibrationResult with dict params {a, b, sigma}
    """
    def vasicek_zero_rate(a: float, b: float, sigma: float, r0: float, t: float) -> float:
        """Vasicek model zero rate."""
        if abs(a) < 1e-10:
            return r0

        B = (1 - math.exp(-a * t)) / a
        A = (B - t) * (a * b - 0.5 * sigma ** 2) / a ** 2 - sigma ** 2 * B ** 2 / (4 * a)

        return (r0 * B - A) / t if t > 0 else r0

    r0 = zero_rates[0] if zero_rates else 0.05

    def residuals(params: List[float]) -> List[float]:
        a, b, sigma = params
        if a <= 0 or sigma <= 0:
            return [1e10] * len(zero_rates)

        errors = []
        for rate, t in zip(zero_rates, times):
            if t > 0:
                model_rate = vasicek_zero_rate(a, b, sigma, r0, t)
                errors.append(model_rate - rate)
        return errors if errors else [0.0]

    optimizer = LevenbergMarquardt(tol=1e-10, maxiter=200)
    result = optimizer.solve(residuals, [initial_a, initial_b, initial_sigma])

    a_opt, b_opt, sigma_opt = result.x

    return CalibrationResult(
        params={"a": a_opt, "b": b_opt, "sigma": sigma_opt, "r0": r0},
        converged=result.converged,
        iterations=result.iterations,
        error=result.residual_norm,
        message=result.message,
    )


# =============================================================================
# Additional Finance Utilities
# =============================================================================

def implied_repo_rate(
    spot: float,
    futures: float,
    time_to_expiry: float,
    income: float = 0.0,
    storage_cost: float = 0.0,
) -> float:
    """
    Compute implied repo rate from spot and futures prices.

    Args:
        spot: Spot price
        futures: Futures price
        time_to_expiry: Time to futures expiry in years
        income: Present value of income (dividends, coupons)
        storage_cost: Present value of storage costs

    Returns:
        Implied repo rate (continuously compounded)

    Example:
        >>> implied_repo_rate(spot=100, futures=102, time_to_expiry=0.5)
        0.0396...
    """
    if time_to_expiry <= 0:
        return 0.0

    # F = (S - I + U) * e^(r*T)
    # r = ln(F / (S - I + U)) / T
    adjusted_spot = spot - income + storage_cost

    if adjusted_spot <= 0:
        raise ValueError("Adjusted spot price must be positive")

    return math.log(futures / adjusted_spot) / time_to_expiry


def basis(
    spot: float,
    futures: float,
) -> float:
    """
    Compute futures basis.

    Args:
        spot: Spot price
        futures: Futures price

    Returns:
        Basis (spot - futures)
    """
    return spot - futures


def carry(
    spot: float,
    futures: float,
    time_to_expiry: float,
) -> float:
    """
    Compute cost of carry (annualized).

    Args:
        spot: Spot price
        futures: Futures price
        time_to_expiry: Time to expiry in years

    Returns:
        Annualized cost of carry
    """
    if time_to_expiry <= 0 or spot <= 0:
        return 0.0

    return math.log(futures / spot) / time_to_expiry


def dv01(
    price_func: Callable[[float], float],
    rate: float,
    bump: float = 0.0001,
) -> float:
    """
    Compute DV01 (dollar value of 1 basis point).

    Args:
        price_func: Function that returns price given rate
        rate: Current rate level
        bump: Bump size (default 1bp)

    Returns:
        DV01 (change in price for 1bp rate increase)
    """
    price_up = price_func(rate + bump)
    price_down = price_func(rate - bump)
    return -(price_up - price_down) / 2


def pv01(
    notional: float,
    discount_factors: List[float],
    times: List[float],
    frequency: int = 2,
) -> float:
    """
    Compute PV01 of a swap (present value of 1bp on fixed leg).

    Args:
        notional: Swap notional
        discount_factors: Discount factors at payment dates
        times: Payment times in years
        frequency: Payment frequency

    Returns:
        PV01
    """
    period = 1.0 / frequency
    annuity = sum(df * period for df in discount_factors)
    return notional * annuity * 0.0001
