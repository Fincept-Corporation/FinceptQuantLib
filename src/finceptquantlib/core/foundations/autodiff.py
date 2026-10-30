"""
Automatic Differentiation Engine

Implements forward-mode automatic differentiation with support for:
- First-order (Dual numbers)
- Higher-order (Hyper-Dual numbers up to 4th order)
- Sparse Jacobian/Hessian computation
- Taylor series expansion

This implementation is designed for maximum performance and numerical stability.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence, Union

__all__ = ["Dual", "HyperDual", "gradient", "hessian", "jacobian", "taylor_expand"]


class Dual:
    """
    Dual number for first-order automatic differentiation.

    A dual number is represented as: f + f' * ε, where ε² = 0
    This enables exact computation of derivatives without numerical errors.

    Attributes:
        value: The function value (real part)
        deriv: The first derivative (dual part)

    Examples:
        >>> x = Dual(2.0, 1.0)  # x with derivative dx/dx = 1
        >>> y = x ** 2 + 3 * x + 1
        >>> print(y.value)  # f(2) = 11
        11.0
        >>> print(y.deriv)  # f'(2) = 2*2 + 3 = 7
        7.0
    """

    __slots__ = ("value", "deriv")

    def __init__(self, value: float, deriv: float = 0.0) -> None:
        """
        Initialize a dual number.

        Args:
            value: Function value
            deriv: First derivative (default: 0.0)
        """
        self.value = float(value)
        self.deriv = float(deriv)

    def __repr__(self) -> str:
        return f"Dual({self.value}, {self.deriv})"

    def __str__(self) -> str:
        return f"{self.value} + {self.deriv}ε"

    # Arithmetic operations

    def __add__(self, other: Union[Dual, float]) -> Dual:
        """Addition: (f + f'ε) + (g + g'ε) = (f+g) + (f'+g')ε"""
        if isinstance(other, Dual):
            return Dual(self.value + other.value, self.deriv + other.deriv)
        return Dual(self.value + other, self.deriv)

    def __radd__(self, other: float) -> Dual:
        return self.__add__(other)

    def __sub__(self, other: Union[Dual, float]) -> Dual:
        """Subtraction: (f + f'ε) - (g + g'ε) = (f-g) + (f'-g')ε"""
        if isinstance(other, Dual):
            return Dual(self.value - other.value, self.deriv - other.deriv)
        return Dual(self.value - other, self.deriv)

    def __rsub__(self, other: float) -> Dual:
        return Dual(other - self.value, -self.deriv)

    def __mul__(self, other: Union[Dual, float]) -> Dual:
        """Multiplication: (f + f'ε)(g + g'ε) = fg + (f'g + fg')ε"""
        if isinstance(other, Dual):
            value = self.value * other.value
            deriv = self.deriv * other.value + self.value * other.deriv
            return Dual(value, deriv)
        return Dual(self.value * other, self.deriv * other)

    def __rmul__(self, other: float) -> Dual:
        return self.__mul__(other)

    def __truediv__(self, other: Union[Dual, float]) -> Dual:
        """Division: (f + f'ε)/(g + g'ε) = f/g + (f'g - fg')/g²ε"""
        if isinstance(other, Dual):
            value = self.value / other.value
            deriv = (self.deriv * other.value - self.value * other.deriv) / (other.value ** 2)
            return Dual(value, deriv)
        return Dual(self.value / other, self.deriv / other)

    def __rtruediv__(self, other: float) -> Dual:
        """Reverse division: c/(f + f'ε) = c/f - cf'/f²ε"""
        value = other / self.value
        deriv = -other * self.deriv / (self.value ** 2)
        return Dual(value, deriv)

    def __pow__(self, exponent: Union[Dual, float]) -> Dual:
        """Power: (f + f'ε)^n = f^n + nf^(n-1)f'ε"""
        if isinstance(exponent, Dual):
            # General case: f^g = exp(g * log(f))
            if self.value <= 0:
                raise ValueError(
                    f"Dual.__pow__: base must be positive when exponent carries "
                    f"derivatives, got base={self.value}"
                )
            value = self.value ** exponent.value
            log_base = math.log(self.value)
            deriv = value * (
                exponent.deriv * log_base +
                exponent.value * self.deriv / self.value
            )
            return Dual(value, deriv)

        n = float(exponent)
        if self.value == 0.0:
            if n >= 1:
                d = n * self.deriv if n == 1 else 0.0
                return Dual(0.0, d)
            raise ValueError(
                f"Dual.__pow__: 0^{n} is undefined for n < 1"
            )
        value = self.value ** n
        deriv = n * (self.value ** (n - 1)) * self.deriv
        return Dual(value, deriv)

    def __rpow__(self, base: float) -> Dual:
        """Reverse power: c^(f + f'ε) = c^f + c^f * ln(c) * f'ε"""
        value = base ** self.value
        deriv = value * math.log(base) * self.deriv
        return Dual(value, deriv)

    def __neg__(self) -> Dual:
        return Dual(-self.value, -self.deriv)

    def __pos__(self) -> Dual:
        return self

    def __abs__(self) -> Dual:
        sign = 1.0 if self.value >= 0 else -1.0
        return Dual(abs(self.value), self.deriv * sign)

    # Comparison operators (compare values only)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Dual):
            return self.value == other.value and self.deriv == other.deriv
        return self.value == other

    def __hash__(self) -> int:
        return hash((self.value, self.deriv))

    def __lt__(self, other: Union[Dual, float]) -> bool:
        return self.value < (other.value if isinstance(other, Dual) else other)

    def __le__(self, other: Union[Dual, float]) -> bool:
        return self.value <= (other.value if isinstance(other, Dual) else other)

    def __gt__(self, other: Union[Dual, float]) -> bool:
        return self.value > (other.value if isinstance(other, Dual) else other)

    def __ge__(self, other: Union[Dual, float]) -> bool:
        return self.value >= (other.value if isinstance(other, Dual) else other)

    def __float__(self) -> float:
        """Convert to float (returns value only)"""
        return self.value

    @staticmethod
    def exp(x: Union[Dual, float]) -> Dual:
        """Exponential function with automatic differentiation."""
        if isinstance(x, Dual):
            return Dual(math.exp(x.value), math.exp(x.value) * x.deriv)
        return Dual(math.exp(x), 0.0)

    @staticmethod
    def sin(x: Union[Dual, float]) -> Dual:
        """Sine function with automatic differentiation."""
        if isinstance(x, Dual):
            return Dual(math.sin(x.value), math.cos(x.value) * x.deriv)
        return Dual(math.sin(x), 0.0)

    @staticmethod
    def cos(x: Union[Dual, float]) -> Dual:
        """Cosine function with automatic differentiation."""
        if isinstance(x, Dual):
            return Dual(math.cos(x.value), -math.sin(x.value) * x.deriv)
        return Dual(math.cos(x), 0.0)

    @staticmethod
    def log(x: Union[Dual, float]) -> Dual:
        """Natural logarithm with automatic differentiation."""
        if isinstance(x, Dual):
            return Dual(math.log(x.value), x.deriv / x.value)
        return Dual(math.log(x), 0.0)


class HyperDual:
    """
    Hyper-Dual number for higher-order automatic differentiation.

    Supports up to 4th order derivatives using the hyper-dual number algebra.
    For nth-order derivatives, uses n infinitesimal units: ε₁, ε₂, ..., εₙ
    where εᵢ² = 0 and εᵢεⱼ ≠ 0 for i ≠ j.

    Attributes:
        value: Function value
        d1: First derivative
        d2: Second derivative
        d3: Third derivative (optional)
        d4: Fourth derivative (optional)
        d12: Cross derivative d²f/dε₁dε₂

    Examples:
        >>> x = HyperDual(2.0, 1.0, 0.0)  # f(x) at x=2
        >>> y = x ** 3
        >>> print(y.value)  # f(2) = 8
        8.0
        >>> print(y.d1)     # f'(2) = 3*4 = 12
        12.0
        >>> print(y.d2)     # f''(2) = 6*2 = 12
        12.0
    """

    __slots__ = ("value", "d1", "d2", "d3", "d4", "d12")

    def __init__(
        self,
        value: float,
        d1: float = 0.0,
        d2: float = 0.0,
        d3: float = 0.0,
        d4: float = 0.0,
        d12: float = 0.0,
    ) -> None:
        """
        Initialize a hyper-dual number.

        Args:
            value: Function value
            d1: First derivative
            d2: Second derivative
            d3: Third derivative
            d4: Fourth derivative
            d12: Cross derivative (for 2D problems)
        """
        self.value = float(value)
        self.d1 = float(d1)
        self.d2 = float(d2)
        self.d3 = float(d3)
        self.d4 = float(d4)
        self.d12 = float(d12)

    def __repr__(self) -> str:
        return f"HyperDual(v={self.value}, d1={self.d1}, d2={self.d2}, d3={self.d3}, d4={self.d4})"

    def __add__(self, other: Union[HyperDual, Dual, float]) -> HyperDual:
        if isinstance(other, HyperDual):
            return HyperDual(
                self.value + other.value,
                self.d1 + other.d1,
                self.d2 + other.d2,
                self.d3 + other.d3,
                self.d4 + other.d4,
                self.d12 + other.d12,
            )
        elif isinstance(other, Dual):
            return HyperDual(
                self.value + other.value,
                self.d1 + other.deriv,
                self.d2,
                self.d3,
                self.d4,
                self.d12,
            )
        return HyperDual(self.value + other, self.d1, self.d2, self.d3, self.d4, self.d12)

    def __radd__(self, other: Union[Dual, float]) -> HyperDual:
        return self.__add__(other)

    def __sub__(self, other: Union[HyperDual, Dual, float]) -> HyperDual:
        if isinstance(other, HyperDual):
            return HyperDual(
                self.value - other.value,
                self.d1 - other.d1,
                self.d2 - other.d2,
                self.d3 - other.d3,
                self.d4 - other.d4,
                self.d12 - other.d12,
            )
        elif isinstance(other, Dual):
            return HyperDual(
                self.value - other.value,
                self.d1 - other.deriv,
                self.d2,
                self.d3,
                self.d4,
                self.d12,
            )
        return HyperDual(self.value - other, self.d1, self.d2, self.d3, self.d4, self.d12)

    def __rsub__(self, other: Union[Dual, float]) -> HyperDual:
        if isinstance(other, Dual):
            return HyperDual(
                other.value - self.value,
                other.deriv - self.d1,
                -self.d2,
                -self.d3,
                -self.d4,
                -self.d12,
            )
        return HyperDual(other - self.value, -self.d1, -self.d2, -self.d3, -self.d4, -self.d12)

    def __mul__(self, other: Union[HyperDual, Dual, float]) -> HyperDual:
        """Product rule with higher-order terms"""
        if isinstance(other, HyperDual):
            value = self.value * other.value
            d1 = self.d1 * other.value + self.value * other.d1
            d2 = (
                self.d2 * other.value +
                2 * self.d1 * other.d1 +
                self.value * other.d2
            )
            d3 = (
                self.d3 * other.value +
                3 * self.d2 * other.d1 +
                3 * self.d1 * other.d2 +
                self.value * other.d3
            )
            d4 = (
                self.d4 * other.value +
                4 * self.d3 * other.d1 +
                6 * self.d2 * other.d2 +
                4 * self.d1 * other.d3 +
                self.value * other.d4
            )
            d12 = self.d12 * other.value + 2 * self.d1 * other.d1 + self.value * other.d12
            return HyperDual(value, d1, d2, d3, d4, d12)
        elif isinstance(other, Dual):
            value = self.value * other.value
            d1 = self.d1 * other.value + self.value * other.deriv
            d2 = self.d2 * other.value + 2 * self.d1 * other.deriv
            d3 = self.d3 * other.value + 3 * self.d2 * other.deriv
            d4 = self.d4 * other.value + 4 * self.d3 * other.deriv
            d12 = self.d12 * other.value + 2 * self.d1 * other.deriv
            return HyperDual(value, d1, d2, d3, d4, d12)
        return HyperDual(
            self.value * other,
            self.d1 * other,
            self.d2 * other,
            self.d3 * other,
            self.d4 * other,
            self.d12 * other,
        )

    def __rmul__(self, other: Union[Dual, float]) -> HyperDual:
        return self.__mul__(other)

    def __truediv__(self, other: Union[HyperDual, Dual, float]) -> HyperDual:
        """Quotient rule with higher-order terms"""
        if isinstance(other, (HyperDual, Dual)):
            # Use f/g = f * g^(-1) and apply chain rule
            return self * (other ** -1)
        return HyperDual(
            self.value / other,
            self.d1 / other,
            self.d2 / other,
            self.d3 / other,
            self.d4 / other,
            self.d12 / other,
        )

    def __rtruediv__(self, other: Union[Dual, float]) -> HyperDual:
        """c / HyperDual = c * HyperDual^(-1)"""
        return other * (self ** -1)

    def __pow__(self, exponent: Union[HyperDual, Dual, float, int]) -> HyperDual:
        """
        Power rule with support for HyperDual/Dual exponents.

        For constant exponent n: uses Faà di Bruno's formula.
        For variable exponent g(x): uses f^g = exp(g * ln(f)).
        """
        if isinstance(exponent, (HyperDual, Dual)):
            # f^g = exp(g * ln(f)) — requires f > 0
            if self.value <= 0:
                raise ValueError(
                    f"HyperDual.__pow__: base must be positive when exponent "
                    f"carries derivatives, got base={self.value}"
                )
            from finceptquantlib.core.utilities.math_utils import exp as ad_exp, log as ad_log
            return ad_exp(exponent * ad_log(self))

        n = float(exponent)

        # Handle zero base
        if self.value == 0.0:
            if n > 4:
                return HyperDual(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            elif n >= 1:
                d1 = n * self.d1 if n >= 1 else 0.0
                return HyperDual(0.0, d1 if n == 1 else 0.0, 0.0, 0.0, 0.0, 0.0)
            raise ValueError(
                f"HyperDual.__pow__: 0^{n} is undefined for n < 1"
            )

        value = self.value ** n

        # First derivative: n * f^(n-1) * f'
        d1 = n * (self.value ** (n - 1)) * self.d1

        # Second derivative: n*(n-1)*f^(n-2)*f'² + n*f^(n-1)*f''
        d2 = (
            n * (n - 1) * (self.value ** (n - 2)) * (self.d1 ** 2) +
            n * (self.value ** (n - 1)) * self.d2
        )

        # Third derivative (Faà di Bruno's formula)
        d3 = (
            n * (n - 1) * (n - 2) * (self.value ** (n - 3)) * (self.d1 ** 3) +
            3 * n * (n - 1) * (self.value ** (n - 2)) * self.d1 * self.d2 +
            n * (self.value ** (n - 1)) * self.d3
        )

        # Fourth derivative
        d4 = (
            n * (n - 1) * (n - 2) * (n - 3) * (self.value ** (n - 4)) * (self.d1 ** 4) +
            6 * n * (n - 1) * (n - 2) * (self.value ** (n - 3)) * (self.d1 ** 2) * self.d2 +
            3 * n * (n - 1) * (self.value ** (n - 2)) * (self.d2 ** 2) +
            4 * n * (n - 1) * (self.value ** (n - 2)) * self.d1 * self.d3 +
            n * (self.value ** (n - 1)) * self.d4
        )

        # Cross derivative
        d12 = (
            n * (self.value ** (n - 1)) * self.d12 +
            n * (n - 1) * (self.value ** (n - 2)) * self.d1 * self.d1
        )

        return HyperDual(value, d1, d2, d3, d4, d12)

    def __rpow__(self, base: float) -> HyperDual:
        """c^f = exp(f * ln(c))"""
        log_base = math.log(base)
        value = base ** self.value
        d1 = value * log_base * self.d1
        d2 = value * log_base * (log_base * self.d1**2 + self.d2)
        d3 = value * log_base * (
            log_base**2 * self.d1**3 +
            3 * log_base * self.d1 * self.d2 +
            self.d3
        )
        d4 = value * log_base * (
            log_base**3 * self.d1**4 +
            6 * log_base**2 * self.d1**2 * self.d2 +
            3 * log_base * self.d2**2 +
            4 * log_base * self.d1 * self.d3 +
            self.d4
        )
        d12 = value * log_base * (log_base * self.d1**2 + self.d12)
        return HyperDual(value, d1, d2, d3, d4, d12)

    def __neg__(self) -> HyperDual:
        return HyperDual(-self.value, -self.d1, -self.d2, -self.d3, -self.d4, -self.d12)

    def __pos__(self) -> HyperDual:
        return self

    def __float__(self) -> float:
        return self.value

    def __abs__(self) -> HyperDual:
        if self.value > 0:
            return HyperDual(self.value, self.d1, self.d2, self.d3, self.d4, self.d12)
        elif self.value < 0:
            return HyperDual(-self.value, -self.d1, -self.d2, -self.d3, -self.d4, -self.d12)
        # At zero, use positive subgradient convention
        return HyperDual(0.0, abs(self.d1), self.d2, self.d3, self.d4, self.d12)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, HyperDual):
            return self.value == other.value and self.d1 == other.d1
        if isinstance(other, Dual):
            return self.value == other.value
        if isinstance(other, (int, float)):
            return self.value == float(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.value, self.d1))

    def __lt__(self, other: Union[HyperDual, Dual, float]) -> bool:
        if isinstance(other, (HyperDual, Dual)):
            return self.value < other.value
        return self.value < float(other)

    def __le__(self, other: Union[HyperDual, Dual, float]) -> bool:
        if isinstance(other, (HyperDual, Dual)):
            return self.value <= other.value
        return self.value <= float(other)

    def __gt__(self, other: Union[HyperDual, Dual, float]) -> bool:
        if isinstance(other, (HyperDual, Dual)):
            return self.value > other.value
        return self.value > float(other)

    def __ge__(self, other: Union[HyperDual, Dual, float]) -> bool:
        if isinstance(other, (HyperDual, Dual)):
            return self.value >= other.value
        return self.value >= float(other)


# Gradient computation functions

def gradient(
    func: Callable[[Union[Dual, float]], Union[Dual, float]],
    x: Union[float, Sequence[float]],
) -> Union[float, list[float]]:
    """
    Compute gradient of a scalar function using automatic differentiation.

    Args:
        func: Function to differentiate (must accept Dual numbers)
        x: Point at which to evaluate gradient (scalar or list)

    Returns:
        Gradient value(s)

    Examples:
        >>> def f(x): return x**2 + 3*x + 1
        >>> gradient(f, 2.0)
        7.0

        >>> def g(x): return x[0]**2 + x[1]**2
        >>> gradient(g, [1.0, 2.0])
        [2.0, 4.0]
    """
    if isinstance(x, (int, float)):
        dual_x = Dual(float(x), 1.0)
        result = func(dual_x)
        return result.deriv if isinstance(result, Dual) else 0.0

    # Vector input - compute gradient component by component
    x_list = [float(v) for v in x]
    grad = []

    for i in range(len(x_list)):
        x_dual = [Dual(x_list[j], 1.0 if i == j else 0.0) for j in range(len(x_list))]
        result = func(x_dual)
        grad.append(result.deriv if isinstance(result, Dual) else 0.0)

    return grad


def hessian(
    func: Callable[[Union[HyperDual, Dual, float]], Union[HyperDual, Dual, float]],
    x: Union[float, Sequence[float]],
) -> Union[float, list[list[float]]]:
    """
    Compute Hessian matrix (second derivatives) using automatic differentiation.

    Args:
        func: Function to differentiate (must accept HyperDual numbers)
        x: Point at which to evaluate Hessian (scalar or list)

    Returns:
        Hessian value or matrix

    Examples:
        >>> def f(x): return x**3
        >>> hessian(f, 2.0)
        12.0
    """
    if isinstance(x, (int, float)):
        hd = HyperDual(float(x), 1.0, 0.0)
        result = func(hd)
        return result.d2 if isinstance(result, HyperDual) else 0.0

    # Vector input - compute Hessian matrix
    x_list = [float(v) for v in x]
    n = len(x_list)
    H = [[0.0] * n for _ in range(n)]

    def second_directional(seeded: tuple[int, ...]) -> float:
        """d²/dt² f(x + t·u) at t=0, where u is 1 on the seeded indices."""
        x_hd = [HyperDual(x_list[k], 1.0 if k in seeded else 0.0) for k in range(n)]
        result = func(x_hd)
        return result.d2 if isinstance(result, HyperDual) else 0.0

    for i in range(n):
        H[i][i] = second_directional((i,))

    # A single HyperDual seed only carries one direction, so the mixed partial is
    # recovered by polarization: D²_{ei+ej} f = H_ii + 2·H_ij + H_jj.
    for i in range(n):
        for j in range(i + 1, n):
            H[i][j] = (second_directional((i, j)) - H[i][i] - H[j][j]) / 2.0
            H[j][i] = H[i][j]

    return H


def jacobian(
    func: Callable[[Sequence[float]], Sequence[float]],
    x: Sequence[float],
) -> list[list[float]]:
    """
    Compute Jacobian matrix using automatic differentiation.

    Args:
        func: Vector-valued function
        x: Point at which to evaluate Jacobian

    Returns:
        Jacobian matrix (m x n) where m = output dim, n = input dim

    Examples:
        >>> def f(x): return [x[0]**2, x[1]**2, x[0]*x[1]]
        >>> jacobian(f, [1.0, 2.0])
        [[2.0, 0.0], [0.0, 4.0], [2.0, 1.0]]
    """
    x_list = [float(v) for v in x]
    n = len(x_list)

    # Evaluate function once to get output dimension
    f_val = func(x_list)
    m = len(f_val) if hasattr(f_val, "__len__") else 1

    J = [[0.0] * n for _ in range(m)]

    for j in range(n):
        # Perturb j-th input
        x_dual = [Dual(x_list[i], 1.0 if i == j else 0.0) for i in range(n)]
        result = func(x_dual)

        if hasattr(result, "__len__"):
            for i in range(m):
                J[i][j] = result[i].deriv if isinstance(result[i], Dual) else 0.0
        else:
            J[0][j] = result.deriv if isinstance(result, Dual) else 0.0

    return J


def taylor_expand(
    func: Callable[[HyperDual], HyperDual],
    x0: float,
    order: int = 4,
) -> list[float]:
    """
    Compute Taylor series coefficients using automatic differentiation.

    Args:
        func: Function to expand
        x0: Expansion point
        order: Maximum order (up to 4)

    Returns:
        List of coefficients [f(x0), f'(x0), f''(x0)/2!, ..., f^(n)(x0)/n!]

    Examples:
        >>> def f(x): return x**3
        >>> taylor_expand(f, 1.0, order=3)  # Matches (x-1)^3 + 3(x-1)^2 + 3(x-1) + 1
        [1.0, 3.0, 3.0, 1.0]
    """
    if order > 4:
        raise ValueError("Maximum supported order is 4")

    hd = HyperDual(x0, 1.0, 0.0, 0.0, 0.0)
    result = func(hd)

    if not isinstance(result, HyperDual):
        return [float(result)]

    coeffs = [result.value]

    if order >= 1:
        coeffs.append(result.d1)
    if order >= 2:
        coeffs.append(result.d2 / 2.0)  # f''(x0) / 2!
    if order >= 3:
        coeffs.append(result.d3 / 6.0)  # f'''(x0) / 3!
    if order >= 4:
        coeffs.append(result.d4 / 24.0)  # f''''(x0) / 4!

    return coeffs
