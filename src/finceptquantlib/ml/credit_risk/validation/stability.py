"""
Population and Characteristic Stability Index

Provides PSI and CSI metrics for detecting model and feature drift
between development (expected) and monitoring (actual) score or
feature distributions. These are standard regulatory metrics for
ongoing model performance monitoring under Basel II/III and SR 11-7.

All operations use pure Python lists without external dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List


@dataclass
class StabilityResult:
    """Result of a stability index calculation.

    Attributes
    ----------
    index : float
        Total stability index (PSI or CSI).
    bin_contributions : List[float]
        Per-bin contribution to the stability index.
    is_stable : bool
        Whether the distribution is considered stable
        (index < threshold).
    threshold : float
        Threshold used for stability determination.
    n_bins : int
        Number of bins used in the calculation.
    """

    index: float
    bin_contributions: List[float]
    is_stable: bool
    threshold: float
    n_bins: int


def _quantile(sorted_values: List[float], q: float) -> float:
    """Compute quantile from a sorted list using linear interpolation.

    Parameters
    ----------
    sorted_values : List[float]
        Pre-sorted values in ascending order.
    q : float
        Quantile in [0, 1].

    Returns
    -------
    float
        Interpolated quantile value.
    """
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _compute_bin_edges(
    values: List[float], n_bins: int
) -> List[float]:
    """Compute quantile-based bin edges from a reference distribution.

    Parameters
    ----------
    values : List[float]
        Reference distribution values.
    n_bins : int
        Number of bins.

    Returns
    -------
    List[float]
        Bin edges of length (n_bins + 1), from -inf to +inf.
    """
    sorted_vals = sorted(values)
    edges: List[float] = [float("-inf")]
    for i in range(1, n_bins):
        q = i / n_bins
        edges.append(_quantile(sorted_vals, q))
    edges.append(float("inf"))
    return edges


def _bin_proportions(
    values: List[float], edges: List[float]
) -> List[float]:
    """Compute the proportion of values falling into each bin.

    Parameters
    ----------
    values : List[float]
        Distribution values to bin.
    edges : List[float]
        Bin edges of length (n_bins + 1).

    Returns
    -------
    List[float]
        Proportion in each bin (sums to 1.0).
    """
    n_bins = len(edges) - 1
    counts = [0] * n_bins
    n = len(values)

    for v in values:
        # Find the bin for this value
        placed = False
        for b in range(n_bins):
            if v <= edges[b + 1]:
                counts[b] += 1
                placed = True
                break
        if not placed:
            counts[n_bins - 1] += 1

    if n == 0:
        return [1.0 / n_bins] * n_bins
    return [c / n for c in counts]


def _compute_psi(
    expected_props: List[float],
    actual_props: List[float],
    eps: float = 1e-10,
) -> tuple[float, List[float]]:
    """Compute PSI from bin proportions.

    Parameters
    ----------
    expected_props : List[float]
        Expected (development) bin proportions.
    actual_props : List[float]
        Actual (monitoring) bin proportions.
    eps : float
        Small constant to avoid log(0) and division by zero.

    Returns
    -------
    tuple[float, List[float]]
        (total_psi, per_bin_contributions).
    """
    contributions: List[float] = []
    for e, a in zip(expected_props, actual_props):
        e_safe = max(e, eps)
        a_safe = max(a, eps)
        psi_i = (a_safe - e_safe) * math.log(a_safe / e_safe)
        contributions.append(psi_i)
    total = sum(contributions)
    return total, contributions


def population_stability_index(
    expected: List[float],
    actual: List[float],
    n_bins: int = 10,
    threshold: float = 0.25,
) -> StabilityResult:
    """Compute Population Stability Index (PSI).

    PSI measures the shift between two score distributions,
    typically the development sample (expected) and a recent
    monitoring sample (actual). It is a standard metric for
    model drift detection in credit risk.

    Interpretation:
    - PSI < 0.10: No significant population change.
    - 0.10 <= PSI < 0.25: Moderate shift, investigation recommended.
    - PSI >= 0.25: Significant shift, model recalibration needed.

    Parameters
    ----------
    expected : List[float]
        Score distribution from the development/training sample.
    actual : List[float]
        Score distribution from the monitoring/validation sample.
    n_bins : int, optional
        Number of quantile-based bins. Default is 10.
    threshold : float, optional
        Stability threshold. Default is 0.25 (significant shift).

    Returns
    -------
    StabilityResult
        PSI value, per-bin contributions, and stability flag.

    Raises
    ------
    ValueError
        If expected or actual is empty, or n_bins < 2.

    Examples
    --------
    >>> from finceptquantlib.ml.credit_risk.validation.stability import (
    ...     population_stability_index,
    ... )
    >>> expected = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    >>> actual = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    >>> result = population_stability_index(expected, actual, n_bins=3)
    >>> result.is_stable
    True
    """
    if len(expected) == 0:
        raise ValueError("expected must not be empty")
    if len(actual) == 0:
        raise ValueError("actual must not be empty")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")

    edges = _compute_bin_edges(expected, n_bins)
    expected_props = _bin_proportions(expected, edges)
    actual_props = _bin_proportions(actual, edges)

    total_psi, contributions = _compute_psi(
        expected_props, actual_props
    )

    return StabilityResult(
        index=total_psi,
        bin_contributions=contributions,
        is_stable=total_psi < threshold,
        threshold=threshold,
        n_bins=n_bins,
    )


def characteristic_stability_index(
    feature_expected: List[float],
    feature_actual: List[float],
    n_bins: int = 10,
    threshold: float = 0.25,
) -> StabilityResult:
    """Compute Characteristic Stability Index (CSI).

    CSI applies the same methodology as PSI but to individual
    feature distributions rather than model scores. It detects
    feature drift that may degrade model performance.

    Interpretation:
    - CSI < 0.10: No significant feature drift.
    - 0.10 <= CSI < 0.25: Moderate drift, monitor closely.
    - CSI >= 0.25: Significant drift, feature may need retraining.

    Parameters
    ----------
    feature_expected : List[float]
        Feature values from the development/training sample.
    feature_actual : List[float]
        Feature values from the monitoring/validation sample.
    n_bins : int, optional
        Number of quantile-based bins. Default is 10.
    threshold : float, optional
        Stability threshold. Default is 0.25.

    Returns
    -------
    StabilityResult
        CSI value, per-bin contributions, and stability flag.

    Raises
    ------
    ValueError
        If feature_expected or feature_actual is empty, or n_bins < 2.

    Examples
    --------
    >>> from finceptquantlib.ml.credit_risk.validation.stability import (
    ...     characteristic_stability_index,
    ... )
    >>> dev = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    >>> mon = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
    >>> result = characteristic_stability_index(dev, mon, n_bins=5)
    >>> result.is_stable
    True
    """
    if len(feature_expected) == 0:
        raise ValueError("feature_expected must not be empty")
    if len(feature_actual) == 0:
        raise ValueError("feature_actual must not be empty")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")

    edges = _compute_bin_edges(feature_expected, n_bins)
    expected_props = _bin_proportions(feature_expected, edges)
    actual_props = _bin_proportions(feature_actual, edges)

    total_csi, contributions = _compute_psi(
        expected_props, actual_props
    )

    return StabilityResult(
        index=total_csi,
        bin_contributions=contributions,
        is_stable=total_csi < threshold,
        threshold=threshold,
        n_bins=n_bins,
    )
