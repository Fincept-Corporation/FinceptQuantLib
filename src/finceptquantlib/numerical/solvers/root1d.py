"""
1D Root Finding

Fast, robust solvers for f(x) = 0:
- brent: Brent's method (recommended default)
- newton: Newton-Raphson with optional derivative
- bisect: Simple bisection
- secant: Secant method
- ridder: Ridder's method
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import math


@dataclass
class RootResult:
    """Result of 1D root finding."""
    root: float
    converged: bool
    iterations: int
    function_value: float


def brent(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-12,
    maxiter: int = 100,
) -> RootResult:
    """
    Brent's method for root finding.

    Combines bisection, secant, and inverse quadratic interpolation.
    Guaranteed to converge if f(a) and f(b) have opposite signs.

    Args:
        f: Function to find root of
        a: Lower bracket
        b: Upper bracket
        tol: Tolerance for convergence
        maxiter: Maximum iterations

    Returns:
        RootResult with root and convergence info

    Raises:
        ValueError: If f(a) and f(b) have same sign
    """
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError(f"f(a)={fa:.6g} and f(b)={fb:.6g} must have opposite signs")

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c, fc = a, fa
    d = b - a
    mflag = True

    for iteration in range(maxiter):
        if abs(fb) < tol:
            return RootResult(b, True, iteration + 1, fb)

        if abs(b - a) < tol:
            return RootResult(b, True, iteration + 1, fb)

        # Try inverse quadratic interpolation
        if abs(fa - fc) > tol and abs(fb - fc) > tol:
            s = (a * fb * fc / ((fa - fb) * (fa - fc)) +
                 b * fa * fc / ((fb - fa) * (fb - fc)) +
                 c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            # Secant method
            s = b - fb * (b - a) / (fb - fa)

        # Conditions for rejecting s and using bisection
        cond1 = not ((3 * a + b) / 4 < s < b or b < s < (3 * a + b) / 4)
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2
        cond3 = not mflag and abs(s - b) >= abs(c - d) / 2
        cond4 = mflag and abs(b - c) < tol
        cond5 = not mflag and abs(c - d) < tol

        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = (a + b) / 2
            mflag = True
        else:
            mflag = False

        fs = f(s)
        d = c
        c, fc = b, fb

        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    return RootResult(b, False, maxiter, fb)


def newton(
    f: Callable[[float], float],
    x0: float,
    fprime: Optional[Callable[[float], float]] = None,
    tol: float = 1e-12,
    maxiter: int = 50,
    h: float = 1e-7,
) -> RootResult:
    """
    Newton-Raphson method for root finding.

    Args:
        f: Function to find root of
        x0: Initial guess
        fprime: Derivative of f (uses finite difference if None)
        tol: Tolerance for convergence
        maxiter: Maximum iterations
        h: Step size for finite difference derivative

    Returns:
        RootResult with root and convergence info
    """
    x = x0

    for iteration in range(maxiter):
        fx = f(x)

        if abs(fx) < tol:
            return RootResult(x, True, iteration + 1, fx)

        # Compute derivative
        if fprime is not None:
            dfx = fprime(x)
        else:
            dfx = (f(x + h) - f(x - h)) / (2 * h)

        if abs(dfx) < 1e-15:
            return RootResult(x, False, iteration + 1, fx)

        x = x - fx / dfx

    fx = f(x)
    return RootResult(x, abs(fx) < tol, maxiter, fx)


def bisect(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-12,
    maxiter: int = 100,
) -> RootResult:
    """
    Bisection method for root finding.

    Simple and robust but slower than Brent.

    Args:
        f: Function to find root of
        a: Lower bracket
        b: Upper bracket
        tol: Tolerance for convergence
        maxiter: Maximum iterations

    Returns:
        RootResult with root and convergence info
    """
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError(f"f(a)={fa:.6g} and f(b)={fb:.6g} must have opposite signs")

    for iteration in range(maxiter):
        c = (a + b) / 2
        fc = f(c)

        if abs(fc) < tol or abs(b - a) < tol:
            return RootResult(c, True, iteration + 1, fc)

        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    c = (a + b) / 2
    return RootResult(c, False, maxiter, f(c))


def secant(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    tol: float = 1e-12,
    maxiter: int = 50,
) -> RootResult:
    """
    Secant method for root finding.

    Like Newton but doesn't require derivative.

    Args:
        f: Function to find root of
        x0: First initial guess
        x1: Second initial guess
        tol: Tolerance for convergence
        maxiter: Maximum iterations

    Returns:
        RootResult with root and convergence info
    """
    f0 = f(x0)
    f1 = f(x1)

    for iteration in range(maxiter):
        if abs(f1) < tol:
            return RootResult(x1, True, iteration + 1, f1)

        if abs(f1 - f0) < 1e-15:
            return RootResult(x1, False, iteration + 1, f1)

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        x0, x1 = x1, x2
        f0, f1 = f1, f(x2)

    return RootResult(x1, abs(f1) < tol, maxiter, f1)


def ridder(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-12,
    maxiter: int = 100,
) -> RootResult:
    """
    Ridder's method for root finding.

    Quadratic convergence with bracketing guarantee.

    Args:
        f: Function to find root of
        a: Lower bracket
        b: Upper bracket
        tol: Tolerance for convergence
        maxiter: Maximum iterations

    Returns:
        RootResult with root and convergence info
    """
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError(f"f(a)={fa:.6g} and f(b)={fb:.6g} must have opposite signs")

    for iteration in range(maxiter):
        c = (a + b) / 2
        fc = f(c)

        if abs(fc) < tol:
            return RootResult(c, True, iteration + 1, fc)

        # Ridder's formula
        disc = fc * fc - fa * fb
        if disc < 0:
            disc = 0.0
        s = math.sqrt(disc)
        if s == 0:
            return RootResult(c, True, iteration + 1, fc)

        dx = (c - a) * fc / s
        if fa - fb < 0:
            dx = -dx

        x = c + dx
        fx = f(x)

        if abs(fx) < tol:
            return RootResult(x, True, iteration + 1, fx)

        # Update bracket
        if fc * fx < 0:
            if c < x:
                a, fa = c, fc
                b, fb = x, fx
            else:
                a, fa = x, fx
                b, fb = c, fc
        elif fa * fx < 0:
            b, fb = x, fx
        else:
            a, fa = x, fx

        if abs(b - a) < tol:
            return RootResult((a + b) / 2, True, iteration + 1, f((a + b) / 2))

    return RootResult((a + b) / 2, False, maxiter, f((a + b) / 2))


# =============================================================================
# Convenience functions with simple API
# =============================================================================

def find_root(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-12,
    maxiter: int = 100,
    method: str = "brent",
    return_info: bool = True,
) -> RootResult | float:
    """
    Find root of f(x) = 0 in interval [a, b].

    Args:
        f: Function to find root of
        a: Lower bracket
        b: Upper bracket
        tol: Tolerance for convergence
        maxiter: Maximum iterations
        method: "brent", "bisect", or "ridder"
        return_info: If False, return just the root value

    Returns:
        RootResult if return_info=True, else float
    """
    if method == "brent":
        result = brent(f, a, b, tol, maxiter)
    elif method == "bisect":
        result = bisect(f, a, b, tol, maxiter)
    elif method == "ridder":
        result = ridder(f, a, b, tol, maxiter)
    else:
        raise ValueError(f"Unknown method: {method}")

    if return_info:
        return result
    return result.root


def find_root_newton(
    f: Callable[[float], float],
    x0: float,
    fprime: Optional[Callable[[float], float]] = None,
    tol: float = 1e-12,
    maxiter: int = 50,
    return_info: bool = True,
) -> RootResult | float:
    """
    Find root using Newton's method.

    Args:
        f: Function to find root of
        x0: Initial guess
        fprime: Optional derivative of f
        tol: Tolerance for convergence
        maxiter: Maximum iterations
        return_info: If False, return just the root value

    Returns:
        RootResult if return_info=True, else float
    """
    result = newton(f, x0, fprime, tol, maxiter)

    if return_info:
        return result
    return result.root
