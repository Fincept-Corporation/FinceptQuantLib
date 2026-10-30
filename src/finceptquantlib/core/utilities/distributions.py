"""
Core Distribution Functions

Standard probability distribution functions used throughout the library.
This module serves as the single source of truth for distribution calculations.

All modules should import from here instead of reimplementing.
"""

from __future__ import annotations

import math


# =============================================================================
# Normal Distribution
# =============================================================================

def norm_cdf(x: float) -> float:
    """
    Standard normal cumulative distribution function.

    Φ(x) = P(Z ≤ x) where Z ~ N(0,1)

    Uses the error function for numerical stability.

    Args:
        x: Value to evaluate CDF at

    Returns:
        Probability P(Z ≤ x)
    """
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def norm_pdf(x: float) -> float:
    """
    Standard normal probability density function.

    φ(x) = (1/√(2π)) * exp(-x²/2)

    Args:
        x: Value to evaluate PDF at

    Returns:
        Density at x
    """
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def norm_ppf(p: float) -> float:
    """
    Inverse of standard normal CDF (quantile/percent point function).

    Returns x such that Φ(x) = p.

    Uses Wichura's AS241 rational approximation for high accuracy.

    Args:
        p: Probability (0 < p < 1)

    Returns:
        Quantile value x
    """
    if p <= 0:
        return float('-inf')
    if p >= 1:
        return float('inf')

    # Rational approximation coefficients (Wichura AS241)
    a = [
        -3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02,
        -3.066479806614716e+01, 2.506628277459239e+00
    ]
    b = [
        -5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01,
        -1.328068155288572e+01
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
        4.374664141464968e+00, 2.938163982698783e+00
    ]
    d = [
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00
    ]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def norm_moment(n: int) -> float:
    """
    n-th moment of standard normal distribution.

    E[Z^n] = 0 for odd n
    E[Z^n] = (n-1)!! for even n

    Args:
        n: Moment order (non-negative integer)

    Returns:
        n-th moment
    """
    if n < 0:
        raise ValueError("Moment order must be non-negative")
    if n % 2 == 1:
        return 0.0
    # Double factorial: (n-1)!! = 1 * 3 * 5 * ... * (n-1)
    result = 1.0
    for k in range(1, n, 2):
        result *= k
    return result


# =============================================================================
# Student-t Distribution
# =============================================================================

def t_pdf(x: float, df: float) -> float:
    """
    Student-t probability density function.

    f(x) = Γ((ν+1)/2) / (√(νπ) Γ(ν/2)) * (1 + x²/ν)^(-(ν+1)/2)

    Args:
        x: Value to evaluate PDF at
        df: Degrees of freedom (ν > 0)

    Returns:
        Density at x
    """
    if df <= 0:
        raise ValueError("Degrees of freedom must be positive")

    log_coef = (
        math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
    )
    return math.exp(log_coef - (df + 1) / 2 * math.log1p(x * x / df))


def t_cdf(x: float, df: float) -> float:
    """
    Student-t cumulative distribution function.

    Uses the exact relation F(x) = 1 - 0.5 * I_{ν/(ν+x²)}(ν/2, 1/2) for x > 0.

    Args:
        x: Value to evaluate CDF at
        df: Degrees of freedom

    Returns:
        Probability P(T ≤ x)
    """
    if df <= 0:
        raise ValueError("Degrees of freedom must be positive")

    if x == 0:
        return 0.5

    tail = 0.5 * _regularized_incomplete_beta(df / 2, 0.5, df / (df + x * x))
    return 1 - tail if x > 0 else tail


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """
    Regularized incomplete beta function I_x(a, b).

    Continued fraction evaluated with the modified Lentz algorithm
    (Numerical Recipes, section 6.4).
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # The continued fraction converges fastest below the mean
    if x > (a + 1) / (a + b + 2):
        return 1 - _regularized_incomplete_beta(b, a, 1 - x)

    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - math.log(a) - log_beta)

    tiny = 1e-300
    f = 1.0
    c = 1.0
    d = 0.0
    for m in range(300):
        m2 = 2 * m

        # Even step: numerator 1 for m == 0, else m(b-m)x / ((a+2m-1)(a+2m))
        num = 1.0 if m == 0 else m * (b - m) * x / ((a + m2 - 1) * (a + m2))
        d = 1 + num * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + num / c
        c = c if abs(c) > tiny else tiny
        f *= d * c

        # Odd step: -(a+m)(a+b+m)x / ((a+2m)(a+2m+1))
        num = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
        d = 1 + num * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + num / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        f *= delta

        if abs(delta - 1) < 1e-14:
            break

    return front * (f - 1)


def t_ppf(p: float, df: float) -> float:
    """
    Inverse of Student-t CDF (quantile/percent point function).

    Returns x such that F_t(x; df) = p.

    Uses Newton's method with normal approximation as starting point.

    Args:
        p: Probability (0 < p < 1)
        df: Degrees of freedom

    Returns:
        Quantile value x
    """
    if p <= 0 or p >= 1:
        if p <= 0:
            return float('-inf')
        return float('inf')
    if df <= 0:
        raise ValueError("Degrees of freedom must be positive")

    # Initial guess from normal approximation
    x = norm_ppf(p)

    # Cornish-Fisher expansion for better initial guess
    g1 = (x ** 3 + x) / (4 * df)
    g2 = (5 * x ** 5 + 16 * x ** 3 + 3 * x) / (96 * df ** 2)
    x = x + g1 + g2

    # Newton's method refinement
    for _ in range(50):
        fx = t_cdf(x, df) - p
        fpx = t_pdf(x, df)
        if abs(fpx) < 1e-30:
            break
        dx = fx / fpx
        x -= dx
        if abs(dx) < 1e-12 * (1 + abs(x)):
            break

    return x


# =============================================================================
# Chi-Squared Distribution
# =============================================================================

def chi2_pdf(x: float, df: float) -> float:
    """
    Chi-squared probability density function.

    f(x) = x^(k/2-1) * e^(-x/2) / (2^(k/2) * Γ(k/2))

    Args:
        x: Value to evaluate PDF at (x ≥ 0)
        df: Degrees of freedom (k > 0)

    Returns:
        Density at x
    """
    if x < 0:
        return 0.0
    if df <= 0:
        raise ValueError("Degrees of freedom must be positive")
    if x == 0:
        if df < 2:
            return float('inf')
        elif df == 2:
            return 0.5
        else:
            return 0.0

    k = df
    return (x ** (k / 2 - 1) * math.exp(-x / 2)) / (2 ** (k / 2) * math.gamma(k / 2))


def chi2_cdf(x: float, df: float) -> float:
    """
    Chi-squared cumulative distribution function.

    Uses the regularized incomplete gamma function.

    Args:
        x: Value to evaluate CDF at
        df: Degrees of freedom

    Returns:
        Probability P(X ≤ x)
    """
    if x <= 0:
        return 0.0
    if df <= 0:
        raise ValueError("Degrees of freedom must be positive")

    # P(X ≤ x) = γ(k/2, x/2) / Γ(k/2)
    # Use series expansion for lower incomplete gamma
    return _lower_incomplete_gamma_ratio(df / 2, x / 2)


def _lower_incomplete_gamma_ratio(a: float, x: float, max_iter: int = 200) -> float:
    """
    Regularized lower incomplete gamma function γ(a,x)/Γ(a).

    Uses series expansion for small x, continued fraction for large x.
    """
    if x < 0:
        return 0.0
    if x == 0:
        return 0.0
    if a <= 0:
        raise ValueError("a must be positive")

    # For x < a + 1, use series expansion
    if x < a + 1:
        term = 1.0 / a
        total = term
        for n in range(1, max_iter):
            term *= x / (a + n)
            total += term
            if abs(term) < 1e-15 * abs(total):
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))

    # For x >= a + 1, use continued fraction for upper incomplete gamma
    # P(a,x) = 1 - Q(a,x)
    cf = _upper_incomplete_gamma_cf(a, x)
    return 1.0 - cf


def _upper_incomplete_gamma_cf(a: float, x: float, max_iter: int = 200) -> float:
    """Upper incomplete gamma using continued fraction."""
    b = x + 1 - a
    c = 1e30
    d = 1 / b
    h = d

    for n in range(1, max_iter):
        an = -n * (n - a)
        b += 2
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break

    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


# =============================================================================
# Gamma Distribution
# =============================================================================

def gamma_pdf(x: float, alpha: float, beta: float = 1.0) -> float:
    """
    Gamma probability density function.

    f(x) = β^α / Γ(α) * x^(α-1) * e^(-βx)

    Args:
        x: Value to evaluate PDF at (x ≥ 0)
        alpha: Shape parameter (α > 0)
        beta: Rate parameter (β > 0)

    Returns:
        Density at x
    """
    if x < 0:
        return 0.0
    if alpha <= 0 or beta <= 0:
        raise ValueError("Shape and rate must be positive")
    if x == 0:
        if alpha < 1:
            return float('inf')
        elif alpha == 1:
            return beta
        else:
            return 0.0

    log_pdf = alpha * math.log(beta) - math.lgamma(alpha) + (alpha - 1) * math.log(x) - beta * x
    return math.exp(log_pdf)


def gamma_cdf(x: float, alpha: float, beta: float = 1.0) -> float:
    """
    Gamma cumulative distribution function.

    Args:
        x: Value to evaluate CDF at
        alpha: Shape parameter
        beta: Rate parameter

    Returns:
        Probability P(X ≤ x)
    """
    if x <= 0:
        return 0.0
    return _lower_incomplete_gamma_ratio(alpha, beta * x)


# =============================================================================
# Exponential Distribution
# =============================================================================

def exp_pdf(x: float, rate: float) -> float:
    """
    Exponential probability density function.

    f(x) = λ * e^(-λx) for x ≥ 0

    Args:
        x: Value to evaluate PDF at
        rate: Rate parameter (λ > 0)

    Returns:
        Density at x
    """
    if x < 0:
        return 0.0
    if rate <= 0:
        raise ValueError("Rate must be positive")
    return rate * math.exp(-rate * x)


def exp_cdf(x: float, rate: float) -> float:
    """
    Exponential cumulative distribution function.

    F(x) = 1 - e^(-λx) for x ≥ 0

    Args:
        x: Value to evaluate CDF at
        rate: Rate parameter

    Returns:
        Probability P(X ≤ x)
    """
    if x < 0:
        return 0.0
    if rate <= 0:
        raise ValueError("Rate must be positive")
    return 1 - math.exp(-rate * x)


def exp_ppf(p: float, rate: float) -> float:
    """
    Exponential quantile function.

    Args:
        p: Probability (0 ≤ p < 1)
        rate: Rate parameter

    Returns:
        Quantile value
    """
    if p < 0 or p >= 1:
        raise ValueError("Probability must be in [0, 1)")
    if rate <= 0:
        raise ValueError("Rate must be positive")
    return -math.log(1 - p) / rate


# =============================================================================
# Multivariate Normal
# =============================================================================

def bivariate_normal_cdf(x: float, y: float, rho: float) -> float:
    """
    Standard bivariate normal CDF.

    P(X ≤ x, Y ≤ y) where (X, Y) ~ N(0, [[1, ρ], [ρ, 1]])

    Uses Drezner-Wesolowsky approximation.

    Args:
        x: First variable threshold
        y: Second variable threshold
        rho: Correlation coefficient (-1 < ρ < 1)

    Returns:
        Joint probability
    """
    if abs(rho) < 1e-10:
        return norm_cdf(x) * norm_cdf(y)

    if rho < 0:
        return max(0, norm_cdf(x) - bivariate_normal_cdf(x, -y, -rho))

    # Drezner-Wesolowsky approximation for positive rho
    a = [0.3253030, 0.4211071, 0.1334425, 0.006374323]
    b = [0.1337764, 0.6243247, 1.3425378, 2.2626645]

    if abs(rho) < 0.9999:
        hk = x * y
        bvn = 0.0

        # Use more Gauss-Legendre points for higher rho
        if abs(rho) < 0.925:
            weights = a
            nodes = b
        else:
            # Extended quadrature for high correlation
            # Gauss-Legendre 6-point quadrature on [0, rho]
            weights = [0.1713245, 0.3607616, 0.4679139,
                       0.4679139, 0.3607616, 0.1713245]
            nodes = [0.0337652, 0.1693953, 0.3806905,
                     0.6193095, 0.8306047, 0.9662348]

        for i in range(len(weights)):
            r = rho * nodes[i]
            rr = 1 - r * r
            if rr > 0:
                bvn += weights[i] * math.exp(
                    (r * hk * 2 - x * x - y * y) / (2 * rr)
                ) / math.sqrt(rr)

        return norm_cdf(x) * norm_cdf(y) + bvn * rho / (2 * math.pi)
    else:
        # Near-perfect correlation
        if rho > 0:
            return min(norm_cdf(x), norm_cdf(y))
        return max(0, norm_cdf(x) + norm_cdf(y) - 1)
