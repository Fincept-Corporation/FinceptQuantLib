"""
Stationarity Transforms and Tests

Provides return computations, differencing operators, and stationarity
testing for financial time-series:

- ``log_returns``: Logarithmic price returns
- ``simple_returns``: Arithmetic price returns
- ``difference``: Integer-order differencing
- ``fractional_difference``: Marcos Lopez de Prado's fractional
  differencing for stationarity while preserving memory
- ``adf_test_statistic``: Augmented Dickey-Fuller unit root test

These are fundamental building blocks for converting non-stationary
price series into stationary features for ML models, following best
practices from *Advances in Financial Machine Learning* (de Prado).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class StationarityTestResult:
    """Result of a stationarity test.

    Attributes
    ----------
    test_statistic : float
        The computed test statistic (e.g., ADF t-statistic).
    p_value_approx : float
        Approximate p-value for the test. For ADF, this is
        interpolated from MacKinnon critical values.
    is_stationary : bool
        Whether the series is stationary at the 5% significance level.
    n_diffs : int
        Number of differences applied (0 if testing raw series).
    method : str
        Name of the test method used.
    """

    test_statistic: float
    p_value_approx: float
    is_stationary: bool
    n_diffs: int
    method: str


def log_returns(prices: List[float]) -> List[float]:
    """Compute logarithmic returns from a price series.

    For a price series p_0, p_1, ..., p_T, the log return at time t is::

        r_t = ln(p_t / p_{t-1})

    Log returns are additive over time and approximately equal to
    simple returns for small values, making them preferred for
    multi-period analysis.

    Parameters
    ----------
    prices : List[float]
        Price series of length T+1. All values must be positive.

    Returns
    -------
    List[float]
        Log returns of length T.

    Raises
    ------
    ValueError
        If any price is non-positive or series has fewer than 2 values.

    Examples
    --------
    >>> log_returns([100.0, 105.0, 110.0])
    [0.04879..., 0.04652...]
    """
    if len(prices) < 2:
        raise ValueError(
            "Price series must have at least 2 observations."
        )
    result: List[float] = []
    for i in range(1, len(prices)):
        if prices[i] <= 0.0 or prices[i - 1] <= 0.0:
            raise ValueError(
                f"Non-positive price at index {i}: "
                f"p[{i}]={prices[i]}, p[{i-1}]={prices[i-1]}. "
                f"Log returns require strictly positive prices."
            )
        result.append(math.log(prices[i] / prices[i - 1]))
    return result


def simple_returns(prices: List[float]) -> List[float]:
    """Compute simple (arithmetic) returns from a price series.

    For a price series p_0, p_1, ..., p_T, the simple return at
    time t is::

        r_t = (p_t - p_{t-1}) / p_{t-1}

    Parameters
    ----------
    prices : List[float]
        Price series of length T+1.

    Returns
    -------
    List[float]
        Simple returns of length T.

    Raises
    ------
    ValueError
        If any denominator price is zero or series is too short.

    Examples
    --------
    >>> simple_returns([100.0, 105.0, 110.0])
    [0.05, 0.04761...]
    """
    if len(prices) < 2:
        raise ValueError(
            "Price series must have at least 2 observations."
        )
    result: List[float] = []
    for i in range(1, len(prices)):
        if prices[i - 1] == 0.0:
            raise ValueError(
                f"Zero price at index {i-1}. Cannot compute return."
            )
        result.append(
            (prices[i] - prices[i - 1]) / prices[i - 1]
        )
    return result


def difference(
    series: List[float], order: int = 1
) -> List[float]:
    """Apply integer-order differencing to a time series.

    First-order differencing: d_t = x_t - x_{t-1}
    Higher orders are applied recursively.

    Parameters
    ----------
    series : List[float]
        Input time series.
    order : int
        Differencing order (d >= 1). Default is 1.

    Returns
    -------
    List[float]
        Differenced series of length len(series) - order.

    Raises
    ------
    ValueError
        If order < 1 or series is too short.

    Examples
    --------
    >>> difference([1.0, 3.0, 6.0, 10.0], order=1)
    [2.0, 3.0, 4.0]
    >>> difference([1.0, 3.0, 6.0, 10.0], order=2)
    [1.0, 1.0]
    """
    if order < 1:
        raise ValueError("Differencing order must be >= 1.")
    if len(series) <= order:
        raise ValueError(
            f"Series length ({len(series)}) must exceed "
            f"differencing order ({order})."
        )
    result = list(series)
    for _ in range(order):
        new_result: List[float] = []
        for i in range(1, len(result)):
            new_result.append(result[i] - result[i - 1])
        result = new_result
    return result


def fractional_difference(
    series: List[float],
    d: float,
    threshold: float = 1e-5,
) -> List[float]:
    """Apply fractional differencing to preserve memory.

    Implements the fixed-width window fractional differencing method
    from Marcos Lopez de Prado's *Advances in Financial Machine
    Learning* (Chapter 5). This transforms a non-stationary series
    to stationarity while retaining as much memory (long-range
    dependence) as possible.

    The fractionally differenced series is computed as::

        y_t = sum_{k=0}^{K} w_k * x_{t-k}

    where the weights w_k are binomial coefficients::

        w_0 = 1
        w_k = -w_{k-1} * (d - k + 1) / k

    Weights are truncated when |w_k| < threshold.

    Parameters
    ----------
    series : List[float]
        Input time series (typically log prices).
    d : float
        Fractional differencing parameter. Values in (0, 1) yield
        partial differencing. d=1.0 gives standard first differencing.
    threshold : float
        Minimum absolute weight for truncation. Weights below this
        value are dropped. Default is 1e-5.

    Returns
    -------
    List[float]
        Fractionally differenced series. The output length equals
        len(series) - K, where K is the number of weights exceeding
        the threshold.

    Raises
    ------
    ValueError
        If d < 0 or series is empty.

    Notes
    -----
    For financial applications, typical values of d are between
    0.3 and 0.7. Lower d preserves more memory but may not achieve
    stationarity; higher d achieves stationarity but loses memory.

    References
    ----------
    .. [1] Lopez de Prado, M. (2018). *Advances in Financial Machine
       Learning*. Wiley. Chapter 5.

    Examples
    --------
    >>> prices = [100.0, 101.0, 102.5, 101.0, 103.0]
    >>> fractional_difference(prices, d=0.5)
    [29.46875]
    """
    if d < 0.0:
        raise ValueError("Fractional order d must be >= 0.")
    if not series:
        raise ValueError("Series must not be empty.")

    # Compute weights using binomial series
    weights: List[float] = [1.0]
    k = 1
    while True:
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        weights.append(w_k)
        k += 1
        # Safety: prevent infinite loops for large d
        if k > len(series):
            break

    window = len(weights)
    n = len(series)

    if window > n:
        # Not enough data for the weight window; return what we can
        # by truncating weights
        weights = weights[:n]
        window = n

    result: List[float] = []
    for t in range(window - 1, n):
        val = 0.0
        for j in range(window):
            val += weights[j] * series[t - j]
        result.append(val)

    return result


def adf_test_statistic(
    series: List[float],
    max_lags: Optional[int] = None,
) -> StationarityTestResult:
    """Augmented Dickey-Fuller unit root test.

    Tests the null hypothesis that the series has a unit root
    (is non-stationary). A sufficiently negative test statistic
    rejects the null in favor of stationarity.

    The test estimates the regression::

        delta_y_t = alpha + gamma * y_{t-1}
                    + sum_{j=1}^{p} beta_j * delta_y_{t-j}
                    + epsilon_t

    and computes the t-statistic for gamma = 0.

    The p-value is approximated by interpolation against MacKinnon
    (1994) critical values for the constant-only case.

    Parameters
    ----------
    series : List[float]
        Time series to test. Must have at least 20 observations
        for meaningful results.
    max_lags : Optional[int]
        Maximum number of lagged differences to include.
        If None, uses int(12 * (n/100)^{1/4}) (Schwert's rule).

    Returns
    -------
    StationarityTestResult
        Test result with statistic, approximate p-value, and
        stationarity decision at the 5% level.

    Raises
    ------
    ValueError
        If series is too short for the test.

    Notes
    -----
    The p-value approximation uses linear interpolation between
    MacKinnon critical values. For production use, consider using
    a dedicated statistics library for exact p-values.

    References
    ----------
    .. [1] Dickey, D.A. and Fuller, W.A. (1979). Distribution of
       the estimators for autoregressive time series with a unit root.
       *Journal of the American Statistical Association*, 74, 427-431.
    .. [2] MacKinnon, J.G. (1994). Approximate asymptotic distribution
       functions for unit-root and cointegration tests.
       *Journal of Business & Economic Statistics*, 12, 167-176.
    """
    n = len(series)
    if n < 7:
        raise ValueError(
            f"Series length ({n}) is too short for ADF test. "
            f"Need at least 7 observations."
        )

    # Determine number of lags using Schwert's rule
    if max_lags is None:
        max_lags = int(12.0 * (n / 100.0) ** 0.25)
    max_lags = min(max_lags, n // 3 - 1)
    if max_lags < 0:
        max_lags = 0

    # First differences
    dy: List[float] = [
        series[i] - series[i - 1] for i in range(1, n)
    ]

    # Build regression matrices for:
    # dy[t] = alpha + gamma * y[t-1] + sum beta_j * dy[t-j] + eps
    # where t runs from max_lags+1 to n-1 (0-indexed in dy)
    start = max_lags
    nobs = len(dy) - start
    if nobs < 3:
        raise ValueError(
            f"Not enough observations ({nobs}) after accounting "
            f"for {max_lags} lags."
        )

    # Number of regressors: intercept + y_{t-1} + max_lags lagged diffs
    n_regressors = 2 + max_lags

    # Build design matrix X and response vector Y
    Y: List[float] = []
    X: List[List[float]] = []
    for t in range(start, len(dy)):
        Y.append(dy[t])
        row: List[float] = [1.0]  # intercept
        row.append(series[t])     # y_{t-1} (level, 0-indexed)
        for j in range(1, max_lags + 1):
            row.append(dy[t - j])
        X.append(row)

    # OLS: beta_hat = (X'X)^{-1} X'Y
    # Build X'X
    p = n_regressors
    XtX: List[List[float]] = [
        [0.0] * p for _ in range(p)
    ]
    XtY: List[float] = [0.0] * p

    for i in range(nobs):
        for a in range(p):
            XtY[a] += X[i][a] * Y[i]
            for b in range(a, p):
                val = X[i][a] * X[i][b]
                XtX[a][b] += val
                if a != b:
                    XtX[b][a] += val

    # Solve via Gaussian elimination with partial pivoting
    beta_hat = _solve_linear_system(XtX, XtY)

    if beta_hat is None:
        # Singular system - return non-stationary result
        return StationarityTestResult(
            test_statistic=0.0,
            p_value_approx=1.0,
            is_stationary=False,
            n_diffs=0,
            method="ADF",
        )

    # Compute residuals and standard error of gamma (index 1)
    residuals: List[float] = []
    for i in range(nobs):
        y_hat = 0.0
        for j in range(p):
            y_hat += beta_hat[j] * X[i][j]
        residuals.append(Y[i] - y_hat)

    # Residual sum of squares
    rss = sum(r * r for r in residuals)
    dof = nobs - p
    if dof <= 0:
        return StationarityTestResult(
            test_statistic=0.0,
            p_value_approx=1.0,
            is_stationary=False,
            n_diffs=0,
            method="ADF",
        )

    sigma2 = rss / dof

    # Standard error of gamma: sqrt(sigma^2 * [(X'X)^{-1}]_{1,1})
    XtX_inv = _invert_matrix(XtX)
    if XtX_inv is None:
        return StationarityTestResult(
            test_statistic=0.0,
            p_value_approx=1.0,
            is_stationary=False,
            n_diffs=0,
            method="ADF",
        )

    se_gamma = math.sqrt(abs(sigma2 * XtX_inv[1][1]))
    if se_gamma < 1e-15:
        return StationarityTestResult(
            test_statistic=0.0,
            p_value_approx=1.0,
            is_stationary=False,
            n_diffs=0,
            method="ADF",
        )

    gamma_hat = beta_hat[1]
    t_stat = gamma_hat / se_gamma

    # Approximate p-value using MacKinnon critical values
    # (constant-only case, interpolated)
    p_value = _mackinnon_p_value(t_stat, nobs)

    return StationarityTestResult(
        test_statistic=t_stat,
        p_value_approx=p_value,
        is_stationary=(p_value < 0.05),
        n_diffs=0,
        method="ADF",
    )


def _solve_linear_system(
    A: List[List[float]], b: List[float]
) -> Optional[List[float]]:
    """Solve Ax = b using Gaussian elimination with partial pivoting.

    Parameters
    ----------
    A : List[List[float]]
        Coefficient matrix (n x n).
    b : List[float]
        Right-hand side vector (n,).

    Returns
    -------
    Optional[List[float]]
        Solution vector, or None if system is singular.
    """
    n = len(A)
    # Augmented matrix
    aug: List[List[float]] = [
        [A[i][j] for j in range(n)] + [b[i]] for i in range(n)
    ]

    for col in range(n):
        # Partial pivoting
        max_row = col
        max_val = abs(aug[col][col])
        for row in range(col + 1, n):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row
        if max_val < 1e-14:
            return None
        if max_row != col:
            aug[col], aug[max_row] = aug[max_row], aug[col]

        # Eliminate below
        pivot = aug[col][col]
        for row in range(col + 1, n):
            factor = aug[row][col] / pivot
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(aug[i][i]) < 1e-14:
            return None
        s = aug[i][n]
        for j in range(i + 1, n):
            s -= aug[i][j] * x[j]
        x[i] = s / aug[i][i]
    return x


def _invert_matrix(
    A: List[List[float]],
) -> Optional[List[List[float]]]:
    """Invert a square matrix using Gauss-Jordan elimination.

    Parameters
    ----------
    A : List[List[float]]
        Square matrix (n x n).

    Returns
    -------
    Optional[List[List[float]]]
        Inverse matrix, or None if singular.
    """
    n = len(A)
    # Augment with identity
    aug: List[List[float]] = []
    for i in range(n):
        row = [A[i][j] for j in range(n)]
        for j in range(n):
            row.append(1.0 if i == j else 0.0)
        aug.append(row)

    for col in range(n):
        # Partial pivoting
        max_row = col
        max_val = abs(aug[col][col])
        for row in range(col + 1, n):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row
        if max_val < 1e-14:
            return None
        if max_row != col:
            aug[col], aug[max_row] = aug[max_row], aug[col]

        # Scale pivot row
        pivot = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pivot

        # Eliminate all other rows
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(2 * n):
                aug[row][j] -= factor * aug[col][j]

    # Extract inverse
    return [
        [aug[i][n + j] for j in range(n)] for i in range(n)
    ]


def _mackinnon_p_value(t_stat: float, nobs: int) -> float:
    """Approximate ADF p-value using MacKinnon critical values.

    Uses linear interpolation between tabulated critical values
    for the constant-only ADF test (no trend).

    Parameters
    ----------
    t_stat : float
        ADF t-statistic.
    nobs : int
        Number of observations used in the regression.

    Returns
    -------
    float
        Approximate p-value in [0, 1].

    Notes
    -----
    The critical values are from MacKinnon (1994) for the
    constant-only specification with a single unit root.
    Asymptotic values are used for large samples.
    """
    # MacKinnon critical values for constant-only ADF test
    # Format: (significance_level, critical_value)
    # Asymptotic values (n -> infinity)
    critical_values = [
        (0.01, -3.43),
        (0.025, -3.12),
        (0.05, -2.86),
        (0.10, -2.57),
        (0.25, -2.07),
        (0.50, -1.53),
        (0.75, -0.97),
        (0.90, -0.44),
        (0.95, -0.07),
        (0.99, 0.65),
    ]

    # Finite-sample correction (approximate)
    # Shift critical values slightly for small samples
    if nobs < 50:
        # Small sample: critical values are more negative
        correction = 1.0 / nobs
        critical_values = [
            (p, cv - correction * 3.0)
            for p, cv in critical_values
        ]
    elif nobs < 200:
        correction = 0.5 / nobs
        critical_values = [
            (p, cv - correction * 3.0)
            for p, cv in critical_values
        ]

    # If t_stat is more negative than the smallest critical value
    if t_stat <= critical_values[0][1]:
        return critical_values[0][0] * 0.5  # very small p-value

    # If t_stat is larger than the largest critical value
    if t_stat >= critical_values[-1][1]:
        return min(1.0, critical_values[-1][0] + 0.005)

    # Linear interpolation
    for i in range(len(critical_values) - 1):
        p_lo, cv_lo = critical_values[i]
        p_hi, cv_hi = critical_values[i + 1]
        if cv_lo <= t_stat <= cv_hi:
            frac = (t_stat - cv_lo) / (cv_hi - cv_lo)
            return p_lo + frac * (p_hi - p_lo)

    # Fallback
    return 0.5
