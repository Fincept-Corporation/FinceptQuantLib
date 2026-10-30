"""
Fast Fourier Transform

FFT implementation for characteristic function pricing and signal processing.

This module provides the single source of truth for FFT algorithms.
"""

from __future__ import annotations

import cmath
import math
from typing import List, Tuple, Callable


def cooley_tukey_fft(x: List[complex]) -> List[complex]:
    """
    Cooley-Tukey FFT algorithm (radix-2, decimation in time).

    Computes the discrete Fourier transform:
    X[k] = Σ_{n=0}^{N-1} x[n] * e^(-2πi*kn/N)

    Input length must be a power of 2.

    Args:
        x: Input sequence (length must be power of 2)

    Returns:
        DFT of input sequence
    """
    n = len(x)

    if n <= 1:
        return x

    # Check if n is power of 2
    if n & (n - 1) != 0:
        raise ValueError(f"Length {n} is not a power of 2")

    # Bit reversal permutation
    result = list(x)
    j = 0
    for i in range(n):
        if i < j:
            result[i], result[j] = result[j], result[i]
        m = n >> 1
        while m >= 1 and j >= m:
            j -= m
            m >>= 1
        j += m

    # Cooley-Tukey butterflies
    length = 2
    while length <= n:
        angle = -2 * cmath.pi / length
        w_n = cmath.exp(complex(0, angle))

        for start in range(0, n, length):
            w = 1
            for k in range(length // 2):
                t = w * result[start + k + length // 2]
                u = result[start + k]
                result[start + k] = u + t
                result[start + k + length // 2] = u - t
                w *= w_n

        length *= 2

    return result


def ifft(x: List[complex]) -> List[complex]:
    """
    Inverse FFT.

    x[n] = (1/N) * Σ_{k=0}^{N-1} X[k] * e^(2πi*kn/N)

    Args:
        x: Input DFT sequence

    Returns:
        Inverse DFT (time domain signal)
    """
    n = len(x)

    # Conjugate, FFT, conjugate, scale
    x_conj = [z.conjugate() for z in x]
    result = cooley_tukey_fft(x_conj)
    return [z.conjugate() / n for z in result]


def fft_real(x: List[float]) -> List[complex]:
    """
    FFT of real-valued input.

    Args:
        x: Real-valued input sequence

    Returns:
        DFT (only positive frequencies are unique due to symmetry)
    """
    return cooley_tukey_fft([complex(xi, 0) for xi in x])


def fft_convolve(a: List[float], b: List[float]) -> List[float]:
    """
    Circular convolution using FFT.

    (a * b)[n] = Σ_k a[k] * b[n-k]

    Args:
        a: First sequence
        b: Second sequence

    Returns:
        Circular convolution
    """
    n = len(a)
    if len(b) != n:
        raise ValueError("Sequences must have same length")

    # Pad to power of 2 for linear convolution
    m = 1
    while m < 2 * n:
        m *= 2

    a_padded = [complex(a[i], 0) if i < n else 0j for i in range(m)]
    b_padded = [complex(b[i], 0) if i < n else 0j for i in range(m)]

    A = cooley_tukey_fft(a_padded)
    B = cooley_tukey_fft(b_padded)

    C = [A[i] * B[i] for i in range(m)]

    c = ifft(C)
    return [c[i].real for i in range(2 * n - 1)]


# =============================================================================
# Pricing-Specific FFT Functions
# =============================================================================

def carr_madan_fft(
    characteristic_fn: Callable[[complex], complex],
    S: float,
    r: float,
    T: float,
    alpha: float = 1.5,
    N: int = 4096,
    eta: float = 0.25
) -> Tuple[List[float], List[float]]:
    """
    Carr-Madan FFT option pricing.

    Computes call prices for a range of strikes using FFT.

    Args:
        characteristic_fn: Characteristic function φ(u) of log(S_T)
        S: Spot price
        r: Risk-free rate
        T: Time to maturity
        alpha: Dampening parameter (typically 1.5)
        N: Number of FFT points (power of 2)
        eta: Grid spacing in Fourier space

    Returns:
        (strikes, call_prices) arrays
    """
    # Grid spacing in log-strike space
    lambda_ = 2 * cmath.pi / (N * eta)

    # Build integrand
    b = N * lambda_ / 2  # Range of log-strikes

    # Characteristic function of modified payoff
    def psi(v: float) -> complex:
        u = complex(v, -(alpha + 1))
        denom = alpha ** 2 + alpha - v ** 2 + complex(0, (2 * alpha + 1) * v)
        if abs(denom) < 1e-10:
            return 0j
        return cmath.exp(-r * T) * characteristic_fn(u) / denom

    # Simpson's rule weights
    simpson = [1.0] * N
    simpson[0] = 1 / 3
    for j in range(1, N):
        simpson[j] = (4 / 3) if j % 2 == 1 else (2 / 3)

    # Build FFT input
    x = []
    for j in range(N):
        v = j * eta
        integrand = psi(v) * cmath.exp(complex(0, v * b))
        x.append(integrand * simpson[j] * eta)

    # FFT
    y = cooley_tukey_fft(x)

    # Extract call prices
    strikes = []
    call_prices = []

    for i in range(N):
        k = -b + lambda_ * i  # Log-strike
        K = math.exp(k)
        strikes.append(K)

        # Call price
        price = (math.exp(-alpha * k) / cmath.pi) * y[i].real
        call_prices.append(max(0, price))

    return strikes, call_prices


def lewis_fft(
    characteristic_fn: Callable[[complex], complex],
    S: float,
    K: float,
    r: float,
    T: float,
    N: int = 4096
) -> float:
    """
    Lewis (2001) single-strike call price by Fourier integration.

    C = e^{-rT} E[S_T] - sqrt(K) e^{-rT} / pi
        * integral_0^inf Re[e^{-iu ln K} phi(u - i/2)] / (u^2 + 1/4) du

    Dividends and other carry are carried by the characteristic function.

    Args:
        characteristic_fn: Characteristic function phi(u) of log(S_T)
        S: Spot price (unused; the spot enters through characteristic_fn)
        K: Strike price
        r: Risk-free rate
        T: Time to maturity
        N: Maximum number of adaptive quadrature subintervals

    Returns:
        Call option price
    """
    from scipy.integrate import quad

    del S  # the spot is encoded in characteristic_fn
    log_k = math.log(K)

    def integrand(u: float) -> float:
        value = cmath.exp(complex(0, -u * log_k)) * characteristic_fn(complex(u, -0.5))
        return value.real / (u * u + 0.25)

    integral = quad(integrand, 0.0, math.inf, limit=N)[0]
    discount = math.exp(-r * T)
    forward = characteristic_fn(complex(0, -1)).real  # E[S_T]
    price = discount * forward - math.sqrt(K) * discount / math.pi * integral
    return max(0.0, price)


def cos_method(
    characteristic_fn: Callable[[complex], complex],
    S: float,
    K: float,
    r: float,
    T: float,
    L: float = 10.0,
    N: int = 256
) -> float:
    """
    COS method for European call pricing (Fang & Oosterlee, 2008).

    The density of y = log(S_T / K) is expanded in a cosine series on a
    truncation range [a, b] centred on its first two cumulants, which are
    estimated numerically from the characteristic function.

    Args:
        characteristic_fn: Characteristic function phi(u) of log(S_T)
        S: Spot price (unused; the spot enters through characteristic_fn)
        K: Strike price
        r: Risk-free rate
        T: Time to maturity
        L: Truncation range in standard deviations (typically 10)
        N: Number of cosine terms

    Returns:
        Call option price
    """
    del S  # the spot is encoded in characteristic_fn
    log_k = math.log(K)

    def phi_y(u: float) -> complex:
        """Characteristic function of y = log(S_T / K)."""
        return characteristic_fn(complex(u, 0)) * cmath.exp(complex(0, -u * log_k))

    # Cumulants of y from finite differences of log(phi_y) at 0
    h = 1e-4
    log_plus = cmath.log(phi_y(h))
    log_minus = cmath.log(phi_y(-h))
    c1 = (log_plus - log_minus).imag / (2 * h)
    c2 = max(-(log_plus + log_minus).real / (h * h), 1e-12)
    a = c1 - L * math.sqrt(c2)
    b = c1 + L * math.sqrt(c2)
    width = b - a

    def chi_k(k: int, c: float, d: float) -> float:
        """Integral of e^y cos(k pi (y - a) / (b - a)) over [c, d]."""
        w = k * math.pi / width
        return (
            math.cos(w * (d - a)) * math.exp(d) - math.cos(w * (c - a)) * math.exp(c)
            + w * (math.sin(w * (d - a)) * math.exp(d) - math.sin(w * (c - a)) * math.exp(c))
        ) / (1 + w * w)

    def psi_k(k: int, c: float, d: float) -> float:
        """Integral of cos(k pi (y - a) / (b - a)) over [c, d]."""
        if k == 0:
            return d - c
        w = k * math.pi / width
        return (math.sin(w * (d - a)) - math.sin(w * (c - a))) / w

    # Call payoff (e^y - 1)^+ in units of K is non-zero on [0, b]
    lower = min(max(0.0, a), b)
    total = 0.0
    for k in range(N):
        u = k * math.pi / width
        v_k = 2.0 / width * (chi_k(k, lower, b) - psi_k(k, lower, b))
        term = (phi_y(u) * cmath.exp(complex(0, -u * a))).real * v_k
        total += 0.5 * term if k == 0 else term

    price = K * math.exp(-r * T) * total
    return max(0.0, price)


# =============================================================================
# Helper Functions
# =============================================================================

def next_power_of_2(n: int) -> int:
    """Return smallest power of 2 >= n."""
    if n <= 0:
        return 1
    p = 1
    while p < n:
        p *= 2
    return p


def pad_to_power_of_2(x: List[complex], fill: complex = 0j) -> List[complex]:
    """Pad sequence to next power of 2."""
    n = len(x)
    m = next_power_of_2(n)
    return x + [fill] * (m - n)
