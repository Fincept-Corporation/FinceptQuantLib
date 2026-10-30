"""
Credit Model Performance Evaluation

Provides comprehensive model performance assessment combining
discrimination metrics, calibration analysis, and goodness-of-fit
tests. Designed for regulatory model validation workflows per
Basel II/III and SR 11-7 requirements.

All computations are pure Python with no external dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from finceptquantlib.ml.credit_risk.credit.discrimination import (
    DiscriminationResult,
    full_discrimination_report,
)


@dataclass
class ModelPerformanceReport:
    """Comprehensive credit model performance report.

    Attributes
    ----------
    discrimination : DiscriminationResult
        Full discrimination metrics (KS, Gini, AUC, etc.).
    brier_score : float
        Brier score = (1/n) * sum((y_i - p_i)^2).
        Measures calibration and discrimination jointly.
    hosmer_lemeshow_stat : float
        Hosmer-Lemeshow chi-squared statistic for
        goodness-of-fit.
    hosmer_lemeshow_pvalue : float
        Approximate p-value for the Hosmer-Lemeshow test.
    calibration_curve : Tuple[List[float], List[float]]
        (mean_predicted, observed_rate) per calibration bin.
    n_samples : int
        Total number of observations.
    n_positives : int
        Number of positive (event/default) observations.
    n_negatives : int
        Number of negative (non-event) observations.
    prevalence : float
        Event rate = n_positives / n_samples.
    """

    discrimination: DiscriminationResult
    brier_score: float
    hosmer_lemeshow_stat: float
    hosmer_lemeshow_pvalue: float
    calibration_curve: Tuple[List[float], List[float]]
    n_samples: int
    n_positives: int
    n_negatives: int
    prevalence: float


def _chi_squared_survival(x: float, df: int) -> float:
    """Approximate chi-squared survival function P(X > x).

    Uses the Wilson-Hilferty normal approximation for the
    chi-squared distribution.

    Parameters
    ----------
    x : float
        Chi-squared test statistic.
    df : int
        Degrees of freedom.

    Returns
    -------
    float
        Approximate p-value (upper tail probability).
    """
    if df <= 0 or x < 0:
        return 1.0

    # Wilson-Hilferty approximation:
    # Z = ((X/df)^(1/3) - (1 - 2/(9*df))) / sqrt(2/(9*df))
    ratio = x / df
    if ratio <= 0:
        return 1.0

    cube_root = ratio ** (1.0 / 3.0)
    correction = 1.0 - 2.0 / (9.0 * df)
    variance = 2.0 / (9.0 * df)

    if variance <= 0:
        return 1.0

    z = (cube_root - correction) / math.sqrt(variance)

    # Standard normal CDF approximation (Abramowitz & Stegun)
    return 1.0 - _norm_cdf(z)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF approximation.

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    float
        P(Z <= x).
    """
    if x < -8.0:
        return 0.0
    if x > 8.0:
        return 1.0

    sign = 1
    if x < 0:
        sign = -1
        x = -x

    p = 0.2316419
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429

    t = 1.0 / (1.0 + p * x)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t

    pdf = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    cdf = 1.0 - pdf * (
        b1 * t + b2 * t2 + b3 * t3 + b4 * t4 + b5 * t5
    )

    if sign == -1:
        cdf = 1.0 - cdf

    return cdf


def evaluate_credit_model(
    y_true: List[int],
    y_score: List[float],
    n_bins: int = 10,
) -> ModelPerformanceReport:
    """Comprehensive credit model performance evaluation.

    Computes discrimination metrics, Brier score, Hosmer-Lemeshow
    goodness-of-fit test, and calibration curve for a binary
    credit scoring model.

    Parameters
    ----------
    y_true : List[int]
        True binary labels (0 = non-default, 1 = default).
    y_score : List[float]
        Predicted probabilities of default in [0, 1].
    n_bins : int
        Number of bins for Hosmer-Lemeshow test and calibration
        curve. Default is 10 (deciles).

    Returns
    -------
    ModelPerformanceReport
        Complete performance report.

    Raises
    ------
    ValueError
        If inputs are invalid.

    Notes
    -----
    The Hosmer-Lemeshow test groups observations into deciles
    of predicted probability, then computes a chi-squared
    statistic comparing observed and expected event counts.
    The test has (n_bins - 2) degrees of freedom.

    A high p-value (> 0.05) suggests the model is well-calibrated.
    """
    if len(y_true) != len(y_score):
        raise ValueError(
            "y_true and y_score must have same length"
        )
    if len(y_true) == 0:
        raise ValueError("Inputs must not be empty")

    n = len(y_true)
    n_pos = sum(y_true)
    n_neg = n - n_pos

    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            "y_true must contain both positive and "
            "negative samples"
        )

    # ---------------------------------------------------------------
    # 1. Discrimination metrics
    # ---------------------------------------------------------------
    disc = full_discrimination_report(y_true, y_score)

    # ---------------------------------------------------------------
    # 2. Brier score
    # ---------------------------------------------------------------
    brier = sum(
        (y - p) ** 2 for y, p in zip(y_true, y_score)
    ) / n

    # ---------------------------------------------------------------
    # 3. Hosmer-Lemeshow test
    # ---------------------------------------------------------------
    # Sort by predicted probability
    paired = sorted(
        zip(y_score, y_true), key=lambda x: x[0]
    )

    # Create approximately equal-sized groups
    bin_size = n // n_bins
    remainder = n % n_bins

    hl_stat = 0.0
    mean_predicted: List[float] = []
    observed_rates: List[float] = []

    idx = 0
    actual_bins = 0
    for b in range(n_bins):
        # Distribute remainder across first bins
        size = bin_size + (1 if b < remainder else 0)
        if size == 0:
            continue

        actual_bins += 1
        bin_data = paired[idx: idx + size]
        idx += size

        # Observed events and expected events
        obs_events = sum(d[1] for d in bin_data)
        exp_prob = sum(d[0] for d in bin_data) / size
        exp_events = sum(d[0] for d in bin_data)

        mean_predicted.append(exp_prob)
        observed_rates.append(obs_events / size)

        # Chi-squared contribution
        # HL = sum((O_g - E_g)^2 / (E_g * (1 - E_g/n_g)))
        # Simplified: sum((O - E)^2 / (n_g * pi_g * (1-pi_g)))
        pi_g = exp_events / size if size > 0 else 0.5
        pi_g = max(1e-10, min(1.0 - 1e-10, pi_g))
        denom = size * pi_g * (1.0 - pi_g)

        if denom > 1e-15:
            hl_stat += (obs_events - exp_events) ** 2 / denom

    # Degrees of freedom = n_groups - 2
    hl_df = max(1, actual_bins - 2)
    hl_pvalue = _chi_squared_survival(hl_stat, hl_df)

    # ---------------------------------------------------------------
    # 4. Assemble report
    # ---------------------------------------------------------------
    prevalence = n_pos / n

    return ModelPerformanceReport(
        discrimination=disc,
        brier_score=brier,
        hosmer_lemeshow_stat=hl_stat,
        hosmer_lemeshow_pvalue=hl_pvalue,
        calibration_curve=(mean_predicted, observed_rates),
        n_samples=n,
        n_positives=n_pos,
        n_negatives=n_neg,
        prevalence=prevalence,
    )
