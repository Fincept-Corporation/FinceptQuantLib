"""
Quadrature Methods

Deterministic numerical integration:
- Simpson's rule
- Trapezoidal rule
- Gauss-Legendre quadrature
- Adaptive quadrature
- Romberg integration
"""

from __future__ import annotations

from typing import Callable


def trapezoid(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int = 100,
) -> float:
    """
    Trapezoidal rule for numerical integration.

    Args:
        f: Function to integrate
        a: Lower bound
        b: Upper bound
        n: Number of intervals

    Returns:
        Approximate integral value
    """
    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))

    for i in range(1, n):
        result += f(a + i * h)

    return result * h


def simpson(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int = 100,
) -> float:
    """
    Simpson's rule for numerical integration.

    Uses composite Simpson's rule for better accuracy than trapezoidal.

    Args:
        f: Function to integrate
        a: Lower bound
        b: Upper bound
        n: Number of intervals (must be even)

    Returns:
        Approximate integral value
    """
    if n % 2 != 0:
        n += 1

    h = (b - a) / n

    result = f(a) + f(b)

    # Odd indices: coefficient 4
    for i in range(1, n, 2):
        result += 4 * f(a + i * h)

    # Even indices: coefficient 2
    for i in range(2, n, 2):
        result += 2 * f(a + i * h)

    return result * h / 3


# Gauss-Legendre nodes and weights (precomputed)
_GL_NODES_WEIGHTS = {
    2: ([-0.5773502691896257, 0.5773502691896257],
        [1.0, 1.0]),
    3: ([-0.7745966692414834, 0.0, 0.7745966692414834],
        [0.5555555555555556, 0.8888888888888888, 0.5555555555555556]),
    4: ([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526],
        [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538]),
    5: ([-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640],
        [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891]),
}


def gauss_legendre(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int = 5,
) -> float:
    """
    Gauss-Legendre quadrature.

    Uses n-point Gaussian quadrature rule.
    Exact for polynomials of degree up to 2n-1.

    Args:
        f: Function to integrate
        a: Lower bound
        b: Upper bound
        n: Number of quadrature points (2-5)

    Returns:
        Approximate integral value
    """
    if n not in _GL_NODES_WEIGHTS:
        raise ValueError(f"n must be 2, 3, 4, or 5, got {n}")

    nodes, weights = _GL_NODES_WEIGHTS[n]

    # Transform from [-1, 1] to [a, b]
    c1 = (b - a) / 2
    c2 = (b + a) / 2

    result = 0.0
    for xi, wi in zip(nodes, weights):
        x = c1 * xi + c2
        result += wi * f(x)

    return result * c1


def adaptive_quad(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_depth: int = 50,
) -> float:
    """
    Adaptive Simpson's quadrature.

    Recursively subdivides intervals to achieve desired accuracy.

    Args:
        f: Function to integrate
        a: Lower bound
        b: Upper bound
        tol: Absolute error tolerance
        max_depth: Maximum recursion depth

    Returns:
        Approximate integral value
    """
    def _adaptive(a: float, b: float, fa: float, fb: float, fc: float, S: float, depth: int) -> float:
        """Recursive adaptive integration."""
        c = (a + b) / 2
        d = (a + c) / 2
        e = (c + b) / 2

        fd = f(d)
        fe = f(e)

        # Simpson's rule on left and right halves
        h = b - a
        S_left = h / 12 * (fa + 4 * fd + fc)
        S_right = h / 12 * (fc + 4 * fe + fb)
        S_new = S_left + S_right

        # Richardson extrapolation estimate
        if depth >= max_depth or abs(S_new - S) < 15 * tol:
            return S_new + (S_new - S) / 15

        # Subdivide
        return (_adaptive(a, c, fa, fc, fd, S_left, depth + 1) +
                _adaptive(c, b, fc, fb, fe, S_right, depth + 1))

    fa = f(a)
    fb = f(b)
    fc = f((a + b) / 2)
    S = (b - a) / 6 * (fa + 4 * fc + fb)

    return _adaptive(a, b, fa, fb, fc, S, 0)


def romberg(
    f: Callable[[float], float],
    a: float,
    b: float,
    max_iter: int = 20,
    tol: float = 1e-10,
) -> float:
    """
    Romberg integration.

    Uses Richardson extrapolation on trapezoidal rule.
    Very accurate for smooth functions.

    Args:
        f: Function to integrate
        a: Lower bound
        b: Upper bound
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Approximate integral value
    """
    # R[i][j] stores the Romberg table
    R = [[0.0] * (max_iter + 1) for _ in range(max_iter + 1)]

    # Initial trapezoidal approximation
    h = b - a
    R[0][0] = 0.5 * h * (f(a) + f(b))

    for i in range(1, max_iter + 1):
        # Trapezoidal rule with 2^i intervals
        h /= 2
        n = 2 ** (i - 1)

        # Add new function evaluations
        s = sum(f(a + (2 * k - 1) * h) for k in range(1, n + 1))
        R[i][0] = 0.5 * R[i - 1][0] + h * s

        # Richardson extrapolation
        for j in range(1, i + 1):
            k = 4 ** j
            R[i][j] = (k * R[i][j - 1] - R[i - 1][j - 1]) / (k - 1)

        # Check convergence
        if i > 0 and abs(R[i][i] - R[i - 1][i - 1]) < tol:
            return R[i][i]

    return R[max_iter][max_iter]
