"""
Model Calibration Metrics

Provides statistical tests and metrics for evaluating the calibration
quality of probability models. Calibration measures whether predicted
probabilities match observed frequencies.

Includes:
- Brier Score and Brier Skill Score
- Hosmer-Lemeshow goodness-of-fit test
- Calibration curve (reliability diagram data)
- Spiegelhalter Z-test for calibration

These metrics are mandated by regulatory frameworks (Basel II/III,
SR 11-7) for credit scoring model validation.

All operations use pure Python lists without external dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class CalibrationReport:
    """Complete calibration analysis result.

    Attributes
    ----------
    brier_score : float
        Brier Score: mean squared error of probability predictions.
    hosmer_lemeshow_statistic : float
        Hosmer-Lemeshow chi-squared test statistic.
    hosmer_lemeshow_p_value : float
        P-value for the Hosmer-Lemeshow test.
    calibration_curve_predicted : List[float]
        Mean predicted probability per bin.
    calibration_curve_observed : List[float]
        Observed event fraction per bin.
    n_bins : int
        Number of bins used for calibration curve.
    spiegelhalter_z : float
        Spiegelhalter Z-test statistic.
    spiegelhalter_p_value : float
        P-value for the Spiegelhalter test.
    """

    brier_score: float
    hosmer_lemeshow_statistic: float
    hosmer_lemeshow_p_value: float
    calibration_curve_predicted: List[float]
    calibration_curve_observed: List[float]
    n_bins: int
    spiegelhalter_z: float
    spiegelhalter_p_value: float


def _validate_calibration_inputs(
    y_true: List[int], y_prob: List[float]
) -> None:
    """Validate calibration metric inputs.

    Parameters
    ----------
    y_true : List[int]
        True binary labels (0 or 1).
    y_prob : List[float]
        Predicted probabilities in [0, 1].

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if len(y_true) != len(y_prob):
        raise ValueError(
            "y_true and y_prob must have the same length"
        )
    if len(y_true) == 0:
        raise ValueError("Inputs must not be empty")
    for val in y_true:
        if val not in (0, 1):
            raise ValueError(
                "y_true must contain only 0 and 1 values"
            )


def _standard_normal_cdf(x: float) -> float:
    """Approximate the standard normal CDF using the error function.

    Uses the Abramowitz and Stegun approximation for erf.
    Maximum error < 1.5e-7.

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    float
        Phi(x) = P(Z <= x) for Z ~ N(0, 1).
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _regularized_gamma_lower(a: float, x: float) -> float:
    """Compute regularized lower incomplete gamma function P(a, x).

    Uses the series expansion for P(a, x):
        P(a, x) = (e^{-x} * x^a / Gamma(a)) * sum_{k=0}^{inf} x^k / (a+1)...(a+k)

    Parameters
    ----------
    a : float
        Shape parameter (must be positive).
    x : float
        Upper integration limit (must be non-negative).

    Returns
    -------
    float
        Regularized lower incomplete gamma P(a, x) in [0, 1].
    """
    if x < 0.0:
        return 0.0
    if x == 0.0:
        return 0.0

    # For large x relative to a, use continued fraction via
    # complement: P(a, x) = 1 - Q(a, x)
    if x > a + 1.0:
        return 1.0 - _regularized_gamma_upper(a, x)

    # Series expansion
    # P(a, x) = e^{-x} * x^a / Gamma(a) * sum_{k=0} x^k / prod(a+1..a+k)
    max_iter = 200
    eps = 1e-12

    # Compute in log space for stability
    log_prefix = -x + a * math.log(x) - math.lgamma(a)

    term = 1.0 / a
    total = term
    for k in range(1, max_iter):
        term *= x / (a + k)
        total += term
        if abs(term) < eps * abs(total):
            break

    return math.exp(log_prefix) * total


def _regularized_gamma_upper(a: float, x: float) -> float:
    """Compute regularized upper incomplete gamma function Q(a, x).

    Uses Lentz's continued fraction algorithm.

    Parameters
    ----------
    a : float
        Shape parameter (must be positive).
    x : float
        Lower integration limit (must be non-negative).

    Returns
    -------
    float
        Regularized upper incomplete gamma Q(a, x) in [0, 1].
    """
    if x < 0.0:
        return 1.0
    if x == 0.0:
        return 1.0

    # Continued fraction via modified Lentz's method
    # Q(a, x) = e^{-x} * x^a / Gamma(a) * CF
    max_iter = 200
    eps = 1e-12
    tiny = 1e-30

    log_prefix = -x + a * math.log(x) - math.lgamma(a)

    # CF: 1/(x+1-a+ 1*(1-a)/(x+3-a+ 2*(2-a)/(x+5-a+ ...)))
    # Using standard form: b0 + a1/(b1 + a2/(b2 + ...))
    # where b0=0, b_i = x + (2i-1) - a, a_i = i*(a-i) for i>=1
    # Actually the continued fraction for Q is:
    # 1/(x - a + 1 + K) where K is the CF

    # Lentz method
    f = tiny
    b0 = x + 1.0 - a
    if abs(b0) < tiny:
        b0 = tiny
    f = b0
    c = b0
    d = 0.0

    for i in range(1, max_iter):
        a_i = -i * (i - a)
        b_i = x + 2.0 * i + 1.0 - a

        d = b_i + a_i * d
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d

        c = b_i + a_i / c
        if abs(c) < tiny:
            c = tiny

        delta = c * d
        f *= delta

        if abs(delta - 1.0) < eps:
            break

    result = math.exp(log_prefix) / f
    return max(0.0, min(1.0, result))


def _chi2_cdf(x: float, df: int) -> float:
    """Compute chi-squared CDF using incomplete gamma function.

    chi2_CDF(x; k) = P(k/2, x/2) where P is the regularized
    lower incomplete gamma function.

    Parameters
    ----------
    x : float
        Chi-squared value (non-negative).
    df : int
        Degrees of freedom (positive integer).

    Returns
    -------
    float
        P(X <= x) for X ~ chi-squared(df).
    """
    if x <= 0.0:
        return 0.0
    if df <= 0:
        return 0.0
    return _regularized_gamma_lower(df / 2.0, x / 2.0)


def brier_score(
    y_true: List[int], y_prob: List[float]
) -> float:
    """Compute Brier Score.

    BS = (1/n) * sum((y_i - p_i)^2)

    The Brier Score measures the mean squared difference between
    predicted probabilities and actual outcomes. Lower is better.

    Decomposition (informational):
    - BS = Reliability - Resolution + Uncertainty
    - Reliability: how close predictions match observed frequencies
    - Resolution: how much predicted probabilities vary
    - Uncertainty: base rate variance (prevalence * (1 - prevalence))

    Parameters
    ----------
    y_true : List[int]
        True binary labels (0 or 1).
    y_prob : List[float]
        Predicted probabilities in [0, 1].

    Returns
    -------
    float
        Brier Score in [0, 1]. Lower is better.

    Raises
    ------
    ValueError
        If inputs are invalid.

    Examples
    --------
    >>> brier_score([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    0.025
    """
    _validate_calibration_inputs(y_true, y_prob)
    n = len(y_true)
    return sum(
        (y_true[i] - y_prob[i]) ** 2 for i in range(n)
    ) / n


def hosmer_lemeshow(
    y_true: List[int],
    y_prob: List[float],
    n_groups: int = 10,
) -> Tuple[float, float]:
    """Compute Hosmer-Lemeshow goodness-of-fit test.

    Groups observations into deciles by predicted probability
    and computes a chi-squared statistic comparing observed and
    expected event counts.

    H0: The model is well-calibrated.
    H1: The model is poorly calibrated.

    Parameters
    ----------
    y_true : List[int]
        True binary labels (0 or 1).
    y_prob : List[float]
        Predicted probabilities in [0, 1].
    n_groups : int, optional
        Number of groups (deciles). Default is 10.

    Returns
    -------
    Tuple[float, float]
        (hl_statistic, p_value) where:
        - hl_statistic: Hosmer-Lemeshow chi-squared statistic.
        - p_value: P-value from chi-squared distribution with
          (n_groups - 2) degrees of freedom.

    Raises
    ------
    ValueError
        If inputs are invalid or n_groups < 3.

    Notes
    -----
    The test statistic is:
        HL = sum_{g=1}^{G} (O_g - E_g)^2 / (n_g * pi_g * (1 - pi_g))

    where O_g is observed events, E_g = n_g * pi_g is expected events,
    n_g is group size, and pi_g is the mean predicted probability.
    """
    _validate_calibration_inputs(y_true, y_prob)
    n = len(y_true)
    if n_groups < 3:
        raise ValueError("n_groups must be at least 3")

    # Sort by predicted probability
    paired = sorted(
        zip(y_prob, y_true), key=lambda x: x[0]
    )

    # Divide into n_groups roughly equal-sized groups
    group_size = n / n_groups
    hl_stat = 0.0
    eps = 1e-10

    for g in range(n_groups):
        start = int(round(g * group_size))
        end = int(round((g + 1) * group_size))
        if start >= end:
            continue

        group = paired[start:end]
        n_g = len(group)
        if n_g == 0:
            continue

        # Observed events
        o_g = sum(label for _, label in group)
        # Mean predicted probability
        pi_g = sum(prob for prob, _ in group) / n_g
        # Expected events
        e_g = n_g * pi_g

        denom = n_g * pi_g * (1.0 - pi_g)
        if abs(denom) < eps:
            # Skip groups with extreme probabilities
            continue

        hl_stat += (o_g - e_g) ** 2 / denom

    df = n_groups - 2
    p_value = 1.0 - _chi2_cdf(hl_stat, df) if df > 0 else 1.0

    return hl_stat, p_value


def calibration_curve(
    y_true: List[int],
    y_prob: List[float],
    n_bins: int = 10,
) -> Tuple[List[float], List[float]]:
    """Compute calibration curve (reliability diagram data).

    Bins predictions into equal-width bins and computes the
    mean predicted probability and observed event fraction
    for each non-empty bin.

    Parameters
    ----------
    y_true : List[int]
        True binary labels (0 or 1).
    y_prob : List[float]
        Predicted probabilities in [0, 1].
    n_bins : int, optional
        Number of equal-width bins. Default is 10.

    Returns
    -------
    Tuple[List[float], List[float]]
        (mean_predicted, fraction_positive) where:
        - mean_predicted: mean predicted probability per bin.
        - fraction_positive: observed event rate per bin.
        Only non-empty bins are returned.

    Raises
    ------
    ValueError
        If inputs are invalid or n_bins < 1.
    """
    _validate_calibration_inputs(y_true, y_prob)
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")

    bin_width = 1.0 / n_bins

    # Accumulate per-bin statistics
    bin_sum_pred: List[float] = [0.0] * n_bins
    bin_sum_true: List[int] = [0] * n_bins
    bin_count: List[int] = [0] * n_bins

    for prob, label in zip(y_prob, y_true):
        # Determine bin index
        bin_idx = int(prob / bin_width)
        # Handle edge case where prob == 1.0
        if bin_idx >= n_bins:
            bin_idx = n_bins - 1
        bin_sum_pred[bin_idx] += prob
        bin_sum_true[bin_idx] += label
        bin_count[bin_idx] += 1

    mean_predicted: List[float] = []
    fraction_positive: List[float] = []

    for b in range(n_bins):
        if bin_count[b] > 0:
            mean_predicted.append(
                bin_sum_pred[b] / bin_count[b]
            )
            fraction_positive.append(
                bin_sum_true[b] / bin_count[b]
            )

    return mean_predicted, fraction_positive


def spiegelhalter_test(
    y_true: List[int], y_prob: List[float]
) -> Tuple[float, float]:
    """Compute Spiegelhalter Z-test for calibration.

    Tests the null hypothesis that the model is well-calibrated.
    The test statistic is:

        Z = sum((y_i - p_i)(1 - 2*p_i)) /
            sqrt(sum(p_i*(1 - p_i)*(1 - 2*p_i)^2))

    Under H0, Z ~ N(0, 1) asymptotically.

    Parameters
    ----------
    y_true : List[int]
        True binary labels (0 or 1).
    y_prob : List[float]
        Predicted probabilities in [0, 1].

    Returns
    -------
    Tuple[float, float]
        (z_statistic, p_value) where:
        - z_statistic: Spiegelhalter Z-test statistic.
        - p_value: Two-sided p-value from N(0, 1).

    Raises
    ------
    ValueError
        If inputs are invalid.

    Notes
    -----
    A small p-value (< 0.05) indicates significant miscalibration.
    """
    _validate_calibration_inputs(y_true, y_prob)
    n = len(y_true)

    numerator = 0.0
    denominator_sq = 0.0

    for i in range(n):
        y_i = y_true[i]
        p_i = y_prob[i]
        residual = y_i - p_i
        one_minus_2p = 1.0 - 2.0 * p_i

        numerator += residual * one_minus_2p
        denominator_sq += (
            p_i * (1.0 - p_i) * one_minus_2p * one_minus_2p
        )

    if denominator_sq <= 0.0:
        return 0.0, 1.0

    z_stat = numerator / math.sqrt(denominator_sq)

    # Two-sided p-value
    p_value = 2.0 * (1.0 - _standard_normal_cdf(abs(z_stat)))

    return z_stat, p_value


def brier_skill_score(
    y_true: List[int], y_prob: List[float]
) -> float:
    """Compute Brier Skill Score.

    BSS = 1 - BS / BS_reference

    where BS_reference = prevalence * (1 - prevalence), the Brier
    Score of a model that always predicts the base rate.

    Parameters
    ----------
    y_true : List[int]
        True binary labels (0 or 1).
    y_prob : List[float]
        Predicted probabilities in [0, 1].

    Returns
    -------
    float
        Brier Skill Score. Values > 0 indicate better performance
        than the reference (naive prevalence) model. Maximum is 1.0.

    Raises
    ------
    ValueError
        If inputs are invalid.

    Examples
    --------
    >>> brier_skill_score([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    0.9
    """
    _validate_calibration_inputs(y_true, y_prob)
    n = len(y_true)
    prevalence = sum(y_true) / n
    bs_ref = prevalence * (1.0 - prevalence)

    if bs_ref == 0.0:
        # All same class: reference model has BS=0
        return 0.0

    bs = brier_score(y_true, y_prob)
    return 1.0 - bs / bs_ref


def full_calibration_report(
    y_true: List[int],
    y_prob: List[float],
    n_bins: int = 10,
    n_groups: int = 10,
) -> CalibrationReport:
    """Generate a comprehensive calibration report.

    Computes all calibration metrics in a single call:
    Brier Score, Hosmer-Lemeshow test, calibration curve,
    and Spiegelhalter Z-test.

    Parameters
    ----------
    y_true : List[int]
        True binary labels (0 or 1).
    y_prob : List[float]
        Predicted probabilities in [0, 1].
    n_bins : int, optional
        Number of bins for calibration curve. Default is 10.
    n_groups : int, optional
        Number of groups for Hosmer-Lemeshow test. Default is 10.

    Returns
    -------
    CalibrationReport
        Complete calibration analysis including all metrics.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    _validate_calibration_inputs(y_true, y_prob)

    bs = brier_score(y_true, y_prob)
    hl_stat, hl_p = hosmer_lemeshow(y_true, y_prob, n_groups)
    cal_pred, cal_obs = calibration_curve(y_true, y_prob, n_bins)
    sp_z, sp_p = spiegelhalter_test(y_true, y_prob)

    return CalibrationReport(
        brier_score=bs,
        hosmer_lemeshow_statistic=hl_stat,
        hosmer_lemeshow_p_value=hl_p,
        calibration_curve_predicted=cal_pred,
        calibration_curve_observed=cal_obs,
        n_bins=n_bins,
        spiegelhalter_z=sp_z,
        spiegelhalter_p_value=sp_p,
    )
