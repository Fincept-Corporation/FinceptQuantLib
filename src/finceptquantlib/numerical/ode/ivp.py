"""
Initial Value Problem (IVP) Solvers

Numerical methods for solving dy/dt = f(t, y), y(t0) = y0:
- Euler: First-order explicit method
- RK4: Classical 4th-order Runge-Kutta
- RK45: Adaptive Runge-Kutta-Fehlberg
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional


@dataclass
class IVPResult:
    """Result of IVP solver."""
    t: List[float]
    y: List[List[float]]  # y[i] is state at time t[i]
    success: bool
    message: str = ""


def euler(
    f: Callable[[float, List[float]], List[float]],
    t_span: Tuple[float, float],
    y0: List[float],
    h: float = 0.01,
) -> IVPResult:
    """
    Euler's method for initial value problems.

    First-order explicit method. Simple but low accuracy.

    Args:
        f: Right-hand side function f(t, y) -> dy/dt
        t_span: (t0, tf) time interval
        y0: Initial condition
        h: Step size

    Returns:
        IVPResult with solution trajectory
    """
    t0, tf = t_span
    t = t0
    y = [float(v) for v in y0]

    t_vals = [t]
    y_vals = [y[:]]

    while t < tf:
        if t + h > tf:
            h = tf - t

        dy = f(t, y)
        y = [yi + h * dyi for yi, dyi in zip(y, dy)]
        t += h

        t_vals.append(t)
        y_vals.append(y[:])

    return IVPResult(t_vals, y_vals, True, "Completed")


def rk4(
    f: Callable[[float, List[float]], List[float]],
    t_span: Tuple[float, float],
    y0: List[float],
    h: float = 0.01,
) -> IVPResult:
    """
    Classical 4th-order Runge-Kutta method.

    Good balance of accuracy and efficiency for smooth problems.

    Args:
        f: Right-hand side function f(t, y) -> dy/dt
        t_span: (t0, tf) time interval
        y0: Initial condition
        h: Step size

    Returns:
        IVPResult with solution trajectory
    """
    t0, tf = t_span
    t = t0
    y = [float(v) for v in y0]
    n = len(y)

    t_vals = [t]
    y_vals = [y[:]]

    while t < tf:
        if t + h > tf:
            h = tf - t

        k1 = f(t, y)
        k2 = f(t + h / 2, [y[i] + h / 2 * k1[i] for i in range(n)])
        k3 = f(t + h / 2, [y[i] + h / 2 * k2[i] for i in range(n)])
        k4 = f(t + h, [y[i] + h * k3[i] for i in range(n)])

        y = [y[i] + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(n)]
        t += h

        t_vals.append(t)
        y_vals.append(y[:])

    return IVPResult(t_vals, y_vals, True, "Completed")


def rk45(
    f: Callable[[float, List[float]], List[float]],
    t_span: Tuple[float, float],
    y0: List[float],
    h_init: float = 0.01,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_step: float = float('inf'),
    h_min: float = 1e-12,
) -> IVPResult:
    """
    Runge-Kutta-Fehlberg 4(5) method with adaptive step size.

    Embedded method using 4th and 5th order estimates for error control.

    Args:
        f: Right-hand side function f(t, y) -> dy/dt
        t_span: (t0, tf) time interval
        y0: Initial condition
        h_init: Initial step size
        rtol: Relative tolerance
        atol: Absolute tolerance
        max_step: Maximum step size
        h_min: Minimum step size

    Returns:
        IVPResult with solution trajectory
    """
    # RKF45 coefficients
    c = [0, 1/4, 3/8, 12/13, 1, 1/2]
    a = [
        [],
        [1/4],
        [3/32, 9/32],
        [1932/2197, -7200/2197, 7296/2197],
        [439/216, -8, 3680/513, -845/4104],
        [-8/27, 2, -3544/2565, 1859/4104, -11/40]
    ]
    b4 = [25/216, 0, 1408/2565, 2197/4104, -1/5, 0]
    b5 = [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]

    t0, tf = t_span
    t = t0
    y = [float(v) for v in y0]
    n = len(y)
    h = min(h_init, max_step)

    t_vals = [t]
    y_vals = [y[:]]

    while t < tf:
        if t + h > tf:
            h = tf - t

        # Compute stages
        k = [None] * 6
        k[0] = f(t, y)

        for i in range(1, 6):
            y_stage = [y[j] + h * sum(a[i][m] * k[m][j] for m in range(i)) for j in range(n)]
            k[i] = f(t + c[i] * h, y_stage)

        # 4th and 5th order estimates
        y4 = [y[j] + h * sum(b4[m] * k[m][j] for m in range(6)) for j in range(n)]
        y5 = [y[j] + h * sum(b5[m] * k[m][j] for m in range(6)) for j in range(n)]

        # Error estimate
        err = max(abs(y5[j] - y4[j]) / (atol + rtol * max(abs(y[j]), abs(y5[j])))
                  for j in range(n))

        if err <= 1.0:
            # Accept step
            t += h
            y = y5[:]
            t_vals.append(t)
            y_vals.append(y[:])

        # Adjust step size
        if err > 0:
            factor = 0.9 * (1.0 / err) ** 0.2
            factor = max(0.1, min(factor, 5.0))
            h = min(max(h * factor, h_min), max_step)
        else:
            h = min(2 * h, max_step)

        if h < h_min:
            return IVPResult(t_vals, y_vals, False, "Step size too small")

    return IVPResult(t_vals, y_vals, True, "Completed")


def solve_ivp(
    f: Callable[[float, List[float]], List[float]],
    t_span: Tuple[float, float],
    y0: List[float],
    method: str = "rk45",
    h: Optional[float] = None,
    **kwargs,
) -> IVPResult:
    """
    Solve initial value problem.

    Convenience function with method selection.

    Args:
        f: Right-hand side function f(t, y) -> dy/dt
        t_span: (t0, tf) time interval
        y0: Initial condition
        method: "euler", "rk4", or "rk45"
        h: Step size (for fixed-step methods)
        **kwargs: Additional arguments for adaptive methods

    Returns:
        IVPResult with solution trajectory
    """
    if method == "euler":
        return euler(f, t_span, y0, h or 0.01)
    elif method == "rk4":
        return rk4(f, t_span, y0, h or 0.01)
    elif method == "rk45":
        return rk45(f, t_span, y0, h_init=h or 0.01, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
