"""
Mathematical utilities with automatic differentiation support.

Implements standard math functions that work with Dual and HyperDual numbers.
"""

from __future__ import annotations

import math
from typing import Union

from finceptquantlib.core.foundations.autodiff import Dual, HyperDual

__all__ = [
    "exp",
    "log",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "power",
    "maximum",
    "minimum",
]


def exp(x: Union[Dual, HyperDual, float]) -> Union[Dual, HyperDual, float]:
    """
    Exponential function with AD support.

    Derivatives: d/dx[e^x] = e^x, d²/dx²[e^x] = e^x, etc.

    Args:
        x: Input (Dual, HyperDual, or float)

    Returns:
        e^x with appropriate derivatives

    Examples:
        >>> x = Dual(0.0, 1.0)
        >>> result = exp(x)
        >>> result.value  # e^0 = 1
        1.0
        >>> result.deriv  # d/dx[e^x]|_{x=0} = 1
        1.0
    """
    if isinstance(x, Dual):
        val = math.exp(x.value)
        return Dual(val, val * x.deriv)

    elif isinstance(x, HyperDual):
        val = math.exp(x.value)
        d1 = val * x.d1
        d2 = val * (x.d2 + x.d1**2)
        d3 = val * (x.d3 + 3 * x.d1 * x.d2 + x.d1**3)
        d4 = val * (x.d4 + 4 * x.d1 * x.d3 + 3 * x.d2**2 + 6 * x.d1**2 * x.d2 + x.d1**4)
        d12 = val * (x.d12 + x.d1 * x.d1)
        return HyperDual(val, d1, d2, d3, d4, d12)

    return math.exp(x)


def log(x: Union[Dual, HyperDual, float]) -> Union[Dual, HyperDual, float]:
    """
    Natural logarithm with AD support.

    Derivatives:
        d/dx[ln(x)] = 1/x
        d²/dx²[ln(x)] = -1/x²
        d³/dx³[ln(x)] = 2/x³
        d⁴/dx⁴[ln(x)] = -6/x⁴

    Args:
        x: Input (must be positive)

    Returns:
        ln(x) with derivatives
    """
    if isinstance(x, Dual):
        val = math.log(x.value)
        return Dual(val, x.deriv / x.value)

    elif isinstance(x, HyperDual):
        val = math.log(x.value)
        x_inv = 1.0 / x.value

        d1 = x.d1 * x_inv
        d2 = (x.d2 - x.d1**2 * x_inv) * x_inv
        d3 = (x.d3 - 3 * x.d1 * x.d2 * x_inv + 2 * x.d1**3 * x_inv**2) * x_inv
        d4 = (
            x.d4 - 4 * x.d1 * x.d3 * x_inv - 3 * x.d2**2 * x_inv +
            12 * x.d1**2 * x.d2 * x_inv**2 - 6 * x.d1**4 * x_inv**3
        ) * x_inv
        d12 = (x.d12 - x.d1 * x.d1 * x_inv) * x_inv

        return HyperDual(val, d1, d2, d3, d4, d12)

    return math.log(x)


def sqrt(x: Union[Dual, HyperDual, float]) -> Union[Dual, HyperDual, float]:
    """
    Square root with AD support.

    Uses x^(1/2) and power rule for derivatives.

    Args:
        x: Input (must be non-negative)

    Returns:
        √x with derivatives
    """
    if isinstance(x, (Dual, HyperDual)):
        return x ** 0.5
    return math.sqrt(x)


def sin(x: Union[Dual, HyperDual, float]) -> Union[Dual, HyperDual, float]:
    """
    Sine function with AD support.

    Derivatives follow cyclic pattern:
        d/dx[sin(x)] = cos(x)
        d²/dx²[sin(x)] = -sin(x)
        d³/dx³[sin(x)] = -cos(x)
        d⁴/dx⁴[sin(x)] = sin(x)

    Args:
        x: Input in radians

    Returns:
        sin(x) with derivatives
    """
    if isinstance(x, Dual):
        val = math.sin(x.value)
        deriv = math.cos(x.value)
        return Dual(val, deriv * x.deriv)

    elif isinstance(x, HyperDual):
        s = math.sin(x.value)
        c = math.cos(x.value)

        d1 = c * x.d1
        d2 = -s * x.d1**2 + c * x.d2
        d3 = -c * x.d1**3 - 3 * s * x.d1 * x.d2 + c * x.d3
        d4 = (
            s * x.d1**4 - 6 * c * x.d1**2 * x.d2 - 3 * s * x.d2**2 -
            4 * s * x.d1 * x.d3 + c * x.d4
        )
        d12 = -s * x.d1 * x.d1 + c * x.d12

        return HyperDual(s, d1, d2, d3, d4, d12)

    return math.sin(x)


def cos(x: Union[Dual, HyperDual, float]) -> Union[Dual, HyperDual, float]:
    """
    Cosine function with AD support.

    Derivatives follow cyclic pattern:
        d/dx[cos(x)] = -sin(x)
        d²/dx²[cos(x)] = -cos(x)
        d³/dx³[cos(x)] = sin(x)
        d⁴/dx⁴[cos(x)] = cos(x)

    Args:
        x: Input in radians

    Returns:
        cos(x) with derivatives
    """
    if isinstance(x, Dual):
        val = math.cos(x.value)
        deriv = -math.sin(x.value)
        return Dual(val, deriv * x.deriv)

    elif isinstance(x, HyperDual):
        s = math.sin(x.value)
        c = math.cos(x.value)

        d1 = -s * x.d1
        d2 = -c * x.d1**2 - s * x.d2
        d3 = s * x.d1**3 - 3 * c * x.d1 * x.d2 - s * x.d3
        d4 = (
            c * x.d1**4 + 6 * s * x.d1**2 * x.d2 - 3 * c * x.d2**2 -
            4 * c * x.d1 * x.d3 - s * x.d4
        )
        d12 = -c * x.d1 * x.d1 - s * x.d12

        return HyperDual(c, d1, d2, d3, d4, d12)

    return math.cos(x)


def tan(x: Union[Dual, HyperDual, float]) -> Union[Dual, HyperDual, float]:
    """
    Tangent function with AD support.

    Uses tan(x) = sin(x)/cos(x) and quotient rule.

    Args:
        x: Input in radians

    Returns:
        tan(x) with derivatives
    """
    if isinstance(x, (Dual, HyperDual)):
        return sin(x) / cos(x)
    return math.tan(x)


def asin(x: Union[Dual, HyperDual, float]) -> Union[Dual, HyperDual, float]:
    """
    Inverse sine (arcsin) with AD support.

    d/dx[asin(x)] = 1 / sqrt(1 - x^2)

    Args:
        x: Input in [-1, 1]

    Returns:
        asin(x) with derivatives
    """
    if isinstance(x, Dual):
        val = math.asin(x.value)
        denom = math.sqrt(1.0 - x.value ** 2)
        return Dual(val, x.deriv / denom)

    elif isinstance(x, HyperDual):
        val = math.asin(x.value)
        v2 = x.value ** 2
        denom = math.sqrt(1.0 - v2)
        denom3 = denom ** 3

        d1 = x.d1 / denom
        d2 = x.d2 / denom + x.value * x.d1 ** 2 / denom3
        d12 = x.d12 / denom + x.value * x.d1 * x.d1 / denom3

        d3 = (
            x.d3 / denom +
            3 * x.value * x.d1 * x.d2 / denom3 +
            x.d1 ** 3 * (1.0 + 2 * v2) / (denom3 * (1.0 - v2))
        )
        d4 = 0.0  # 4th order omitted for complexity; rarely needed for asin

        return HyperDual(val, d1, d2, d3, d4, d12)

    return math.asin(x)


def acos(x: Union[Dual, HyperDual, float]) -> Union[Dual, HyperDual, float]:
    """
    Inverse cosine (arccos) with AD support.

    d/dx[acos(x)] = -1 / sqrt(1 - x^2)

    Args:
        x: Input in [-1, 1]

    Returns:
        acos(x) with derivatives
    """
    if isinstance(x, (Dual, HyperDual)):
        # acos(x) = pi/2 - asin(x)
        result = asin(x)
        if isinstance(result, Dual):
            return Dual(math.pi / 2 - result.value, -result.deriv)
        return HyperDual(
            math.pi / 2 - result.value,
            -result.d1, -result.d2, -result.d3, -result.d4, -result.d12
        )

    return math.acos(x)


def atan(x: Union[Dual, HyperDual, float]) -> Union[Dual, HyperDual, float]:
    """
    Inverse tangent (arctan) with AD support.

    d/dx[atan(x)] = 1 / (1 + x^2)

    Args:
        x: Input

    Returns:
        atan(x) with derivatives
    """
    if isinstance(x, Dual):
        val = math.atan(x.value)
        denom = 1.0 + x.value ** 2
        return Dual(val, x.deriv / denom)

    elif isinstance(x, HyperDual):
        val = math.atan(x.value)
        v = x.value
        denom = 1.0 + v ** 2
        denom2 = denom ** 2

        d1 = x.d1 / denom
        d2 = x.d2 / denom - 2 * v * x.d1 ** 2 / denom2
        d12 = x.d12 / denom - 2 * v * x.d1 * x.d1 / denom2

        d3 = (
            x.d3 / denom -
            6 * v * x.d1 * x.d2 / denom2 +
            x.d1 ** 3 * (6 * v ** 2 - 2) / (denom ** 3)
        )
        d4 = 0.0  # 4th order omitted for complexity

        return HyperDual(val, d1, d2, d3, d4, d12)

    return math.atan(x)


def sinh(x: Union[Dual, HyperDual, float]) -> Union[Dual, HyperDual, float]:
    """
    Hyperbolic sine with AD support.

    Derivatives:
        d/dx[sinh(x)] = cosh(x)
        d²/dx²[sinh(x)] = sinh(x)

    Args:
        x: Input

    Returns:
        sinh(x) with derivatives
    """
    if isinstance(x, Dual):
        val = math.sinh(x.value)
        deriv = math.cosh(x.value)
        return Dual(val, deriv * x.deriv)

    elif isinstance(x, HyperDual):
        s = math.sinh(x.value)
        c = math.cosh(x.value)

        d1 = c * x.d1
        d2 = s * x.d1**2 + c * x.d2
        d3 = c * x.d1**3 + 3 * s * x.d1 * x.d2 + c * x.d3
        d4 = (
            s * x.d1**4 + 6 * c * x.d1**2 * x.d2 + 3 * s * x.d2**2 +
            4 * s * x.d1 * x.d3 + c * x.d4
        )
        d12 = s * x.d1 * x.d1 + c * x.d12

        return HyperDual(s, d1, d2, d3, d4, d12)

    return math.sinh(x)


def cosh(x: Union[Dual, HyperDual, float]) -> Union[Dual, HyperDual, float]:
    """
    Hyperbolic cosine with AD support.

    Derivatives:
        d/dx[cosh(x)] = sinh(x)
        d²/dx²[cosh(x)] = cosh(x)

    Args:
        x: Input

    Returns:
        cosh(x) with derivatives
    """
    if isinstance(x, Dual):
        val = math.cosh(x.value)
        deriv = math.sinh(x.value)
        return Dual(val, deriv * x.deriv)

    elif isinstance(x, HyperDual):
        s = math.sinh(x.value)
        c = math.cosh(x.value)

        d1 = s * x.d1
        d2 = c * x.d1**2 + s * x.d2
        d3 = s * x.d1**3 + 3 * c * x.d1 * x.d2 + s * x.d3
        d4 = (
            c * x.d1**4 + 6 * s * x.d1**2 * x.d2 + 3 * c * x.d2**2 +
            4 * c * x.d1 * x.d3 + s * x.d4
        )
        d12 = c * x.d1 * x.d1 + s * x.d12

        return HyperDual(c, d1, d2, d3, d4, d12)

    return math.cosh(x)


def tanh(x: Union[Dual, HyperDual, float]) -> Union[Dual, HyperDual, float]:
    """
    Hyperbolic tangent with AD support.

    Uses tanh(x) = sinh(x)/cosh(x).

    Args:
        x: Input

    Returns:
        tanh(x) with derivatives
    """
    if isinstance(x, (Dual, HyperDual)):
        return sinh(x) / cosh(x)
    return math.tanh(x)


def power(x: Union[Dual, HyperDual, float], n: float) -> Union[Dual, HyperDual, float]:
    """
    Power function x^n with AD support.

    Args:
        x: Base
        n: Exponent

    Returns:
        x^n with derivatives
    """
    if isinstance(x, (Dual, HyperDual)):
        return x ** n
    return x ** n


def maximum(
    x: Union[Dual, HyperDual, float],
    y: Union[Dual, HyperDual, float]
) -> Union[Dual, HyperDual, float]:
    """
    Element-wise maximum with AD support.

    Note: Non-smooth at x = y, derivative is undefined at that point.
    We use the convention: d/dx[max(x,y)] = 1 if x >= y, else 0.

    Args:
        x: First argument
        y: Second argument

    Returns:
        max(x, y) with derivatives
    """
    if isinstance(x, Dual):
        x_val = x.value
        y_val = y.value if isinstance(y, Dual) else y

        if x_val >= y_val:
            return x
        return y if isinstance(y, Dual) else Dual(y, 0.0)

    elif isinstance(x, HyperDual):
        x_val = x.value
        y_val = y.value if isinstance(y, HyperDual) else y

        if x_val >= y_val:
            return x
        return y if isinstance(y, HyperDual) else HyperDual(y, 0.0, 0.0, 0.0, 0.0, 0.0)

    return max(x, y)


def minimum(
    x: Union[Dual, HyperDual, float],
    y: Union[Dual, HyperDual, float]
) -> Union[Dual, HyperDual, float]:
    """
    Element-wise minimum with AD support.

    Note: Non-smooth at x = y, derivative is undefined at that point.

    Args:
        x: First argument
        y: Second argument

    Returns:
        min(x, y) with derivatives
    """
    if isinstance(x, Dual):
        x_val = x.value
        y_val = y.value if isinstance(y, Dual) else y

        if x_val <= y_val:
            return x
        return y if isinstance(y, Dual) else Dual(y, 0.0)

    elif isinstance(x, HyperDual):
        x_val = x.value
        y_val = y.value if isinstance(y, HyperDual) else y

        if x_val <= y_val:
            return x
        return y if isinstance(y, HyperDual) else HyperDual(y, 0.0, 0.0, 0.0, 0.0, 0.0)

    return min(x, y)
