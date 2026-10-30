"""
Credit Model Discrimination Metrics

Provides functions for evaluating the discriminatory power of credit
scoring models, including KS statistic, Gini coefficient, AUC-ROC,
CAP curve, and accuracy ratio. All functions operate on plain Python
lists without external dependencies.

These metrics are standard in credit risk model validation per
Basel II/III regulatory requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class DiscriminationResult:
    """Complete discrimination analysis result.

    Attributes
    ----------
    ks_statistic : float
        Kolmogorov-Smirnov statistic (max separation between
        cumulative event/non-event distributions).
    ks_threshold : float
        Score threshold at which KS statistic is achieved.
    gini_coefficient : float
        Gini coefficient = 2 * AUC - 1.
    auc_roc : float
        Area Under the Receiver Operating Characteristic curve.
    accuracy_ratio : float
        Accuracy Ratio from CAP analysis.
    roc_curve : Tuple[List[float], List[float]]
        (FPR, TPR) points of the ROC curve.
    cap_curve : Tuple[List[float], List[float]]
        (cumulative_population, cumulative_events) of the CAP curve.
    """

    ks_statistic: float
    ks_threshold: float
    gini_coefficient: float
    auc_roc: float
    accuracy_ratio: float
    roc_curve: Tuple[List[float], List[float]]
    cap_curve: Tuple[List[float], List[float]]


def _validate_inputs(
    y_true: List[int], y_score: List[float]
) -> None:
    """Validate discrimination metric inputs.

    Parameters
    ----------
    y_true : List[int]
        True binary labels (0 or 1).
    y_score : List[float]
        Predicted scores or probabilities.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if len(y_true) != len(y_score):
        raise ValueError(
            "y_true and y_score must have the same length"
        )
    if len(y_true) == 0:
        raise ValueError("Inputs must not be empty")
    unique_labels = set(y_true)
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            "y_true must contain only 0 and 1 values"
        )
    if 0 not in unique_labels or 1 not in unique_labels:
        raise ValueError(
            "y_true must contain both positive and negative samples"
        )


def roc_curve(
    y_true: List[int], y_score: List[float]
) -> Tuple[List[float], List[float]]:
    """Compute Receiver Operating Characteristic (ROC) curve.

    Sorts samples by descending score and computes TPR and FPR
    at each distinct threshold.

    Parameters
    ----------
    y_true : List[int]
        True binary labels (0 or 1).
    y_score : List[float]
        Predicted scores (higher = more likely positive).

    Returns
    -------
    Tuple[List[float], List[float]]
        (fpr_list, tpr_list) points defining the ROC curve,
        starting at (0, 0) and ending at (1, 1).

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    _validate_inputs(y_true, y_score)

    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos

    # Sort by score descending, break ties by putting negatives first
    # so that tied scores don't inflate TPR
    paired = sorted(
        zip(y_score, y_true),
        key=lambda x: (-x[0], x[1]),
    )

    fpr_list: List[float] = [0.0]
    tpr_list: List[float] = [0.0]

    tp = 0
    fp = 0

    i = 0
    n = len(paired)
    while i < n:
        # Process all samples with the same score
        current_score = paired[i][0]
        while i < n and paired[i][0] == current_score:
            if paired[i][1] == 1:
                tp += 1
            else:
                fp += 1
            i += 1

        fpr_list.append(fp / n_neg)
        tpr_list.append(tp / n_pos)

    # Ensure we end at (1, 1)
    if fpr_list[-1] != 1.0 or tpr_list[-1] != 1.0:
        fpr_list.append(1.0)
        tpr_list.append(1.0)

    return fpr_list, tpr_list


def auc_roc(
    y_true: List[int], y_score: List[float]
) -> float:
    """Compute Area Under the ROC Curve using trapezoidal rule.

    Parameters
    ----------
    y_true : List[int]
        True binary labels (0 or 1).
    y_score : List[float]
        Predicted scores (higher = more likely positive).

    Returns
    -------
    float
        AUC-ROC value in [0, 1]. A value of 0.5 indicates
        random performance; 1.0 indicates perfect discrimination.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    fpr_list, tpr_list = roc_curve(y_true, y_score)

    # Trapezoidal integration
    auc = 0.0
    for i in range(1, len(fpr_list)):
        dx = fpr_list[i] - fpr_list[i - 1]
        auc += dx * (tpr_list[i] + tpr_list[i - 1]) / 2.0

    return auc


def gini_coefficient(
    y_true: List[int], y_score: List[float]
) -> float:
    """Compute Gini coefficient from AUC-ROC.

    Gini = 2 * AUC - 1

    The Gini coefficient measures the discriminatory power of a
    model. Values range from 0 (random) to 1 (perfect).

    Parameters
    ----------
    y_true : List[int]
        True binary labels (0 or 1).
    y_score : List[float]
        Predicted scores (higher = more likely positive).

    Returns
    -------
    float
        Gini coefficient in [0, 1] for a reasonable model.
    """
    return 2.0 * auc_roc(y_true, y_score) - 1.0


def ks_statistic(
    y_true: List[int], y_score: List[float]
) -> Tuple[float, float]:
    """Compute Kolmogorov-Smirnov statistic.

    Measures the maximum separation between the cumulative
    distribution functions of event and non-event scores.

    Parameters
    ----------
    y_true : List[int]
        True binary labels (0 or 1).
    y_score : List[float]
        Predicted scores (higher = more likely positive).

    Returns
    -------
    Tuple[float, float]
        (ks_value, optimal_threshold) where ks_value is the
        maximum separation and optimal_threshold is the score
        at which it occurs.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    _validate_inputs(y_true, y_score)

    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos

    # Sort by score descending
    paired = sorted(
        zip(y_score, y_true), key=lambda x: -x[0]
    )

    cum_pos = 0
    cum_neg = 0
    max_ks = 0.0
    best_threshold = paired[0][0]

    i = 0
    n = len(paired)
    while i < n:
        current_score = paired[i][0]
        while i < n and paired[i][0] == current_score:
            if paired[i][1] == 1:
                cum_pos += 1
            else:
                cum_neg += 1
            i += 1

        ks_val = abs(cum_pos / n_pos - cum_neg / n_neg)
        if ks_val > max_ks:
            max_ks = ks_val
            best_threshold = current_score

    return max_ks, best_threshold


def cap_curve(
    y_true: List[int], y_score: List[float]
) -> Tuple[List[float], List[float]]:
    """Compute Cumulative Accuracy Profile (CAP) curve.

    The CAP curve plots the cumulative proportion of events
    captured against the cumulative proportion of the population,
    when sorted by model score (descending).

    Parameters
    ----------
    y_true : List[int]
        True binary labels (0 or 1).
    y_score : List[float]
        Predicted scores (higher = more likely positive).

    Returns
    -------
    Tuple[List[float], List[float]]
        (cumulative_population, cumulative_events) points,
        starting at (0, 0) and ending at (1, 1).

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    _validate_inputs(y_true, y_score)

    n = len(y_true)
    n_pos = sum(y_true)

    # Sort by score descending
    paired = sorted(
        zip(y_score, y_true), key=lambda x: -x[0]
    )

    cum_pop: List[float] = [0.0]
    cum_events: List[float] = [0.0]

    running_events = 0
    for i, (score, label) in enumerate(paired):
        running_events += label
        cum_pop.append((i + 1) / n)
        cum_events.append(running_events / n_pos)

    return cum_pop, cum_events


def accuracy_ratio(
    y_true: List[int], y_score: List[float]
) -> float:
    """Compute Accuracy Ratio from CAP analysis.

    AR = (area between model CAP and random) /
         (area between perfect CAP and random)

    The random model CAP is the diagonal. The perfect model
    captures all events in the first n_pos / n samples.

    Parameters
    ----------
    y_true : List[int]
        True binary labels (0 or 1).
    y_score : List[float]
        Predicted scores (higher = more likely positive).

    Returns
    -------
    float
        Accuracy ratio in [0, 1]. Equivalent to Gini coefficient.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    _validate_inputs(y_true, y_score)

    n = len(y_true)
    n_pos = sum(y_true)

    cum_pop, cum_events = cap_curve(y_true, y_score)

    # Model area (trapezoidal rule) minus random area (0.5)
    model_area = 0.0
    for i in range(1, len(cum_pop)):
        dx = cum_pop[i] - cum_pop[i - 1]
        model_area += dx * (cum_events[i] + cum_events[i - 1]) / 2.0

    random_area = 0.5  # diagonal

    # Perfect model area: captures all events in first n_pos/n
    # of the population, then stays at 1.0
    prevalence = n_pos / n
    # Area = triangle up to prevalence + rectangle after
    perfect_area = (prevalence * 1.0 / 2.0) + (1.0 - prevalence) * 1.0

    if abs(perfect_area - random_area) < 1e-15:
        return 0.0

    return (model_area - random_area) / (perfect_area - random_area)


def full_discrimination_report(
    y_true: List[int], y_score: List[float]
) -> DiscriminationResult:
    """Compute all discrimination metrics in one pass.

    Generates a comprehensive discrimination analysis including
    KS statistic, Gini coefficient, AUC-ROC, accuracy ratio,
    and both ROC and CAP curves.

    Parameters
    ----------
    y_true : List[int]
        True binary labels (0 or 1).
    y_score : List[float]
        Predicted scores (higher = more likely positive).

    Returns
    -------
    DiscriminationResult
        Complete discrimination analysis with all metrics.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    _validate_inputs(y_true, y_score)

    roc = roc_curve(y_true, y_score)
    auc = auc_roc(y_true, y_score)
    gini = 2.0 * auc - 1.0
    ks_val, ks_thresh = ks_statistic(y_true, y_score)
    cap = cap_curve(y_true, y_score)
    ar = accuracy_ratio(y_true, y_score)

    return DiscriminationResult(
        ks_statistic=ks_val,
        ks_threshold=ks_thresh,
        gini_coefficient=gini,
        auc_roc=auc,
        accuracy_ratio=ar,
        roc_curve=roc,
        cap_curve=cap,
    )
