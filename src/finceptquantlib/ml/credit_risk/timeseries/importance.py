"""
Time Series Feature Importance

Provides permutation-based feature importance and sequential
feature selection with walk-forward cross-validation. Designed
for temporal data where standard random shuffling would destroy
temporal structure.

All computations are pure Python.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol, Tuple



# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

class _Fittable(Protocol):
    """Protocol for models that support fit and predict."""

    def fit(
        self, X: List[List[float]], y: List[float]
    ) -> object: ...

    def predict(self, X: List[List[float]]) -> object: ...


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TimeSeriesImportanceResult:
    """Result of time series feature importance analysis.

    Attributes
    ----------
    feature_names : List[str]
        Feature names sorted by importance (descending).
    importances : List[float]
        Mean importance scores corresponding to ``feature_names``.
    importances_std : List[float]
        Standard deviation of importance scores across repeats.
    baseline_score : float
        Model performance score without any permutation.
    """

    feature_names: List[str]
    importances: List[float]
    importances_std: List[float]
    baseline_score: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_predictions(result: object) -> List[float]:
    """Extract prediction list from a model result object.

    Parameters
    ----------
    result : object
        Either a PredictResult with ``.predictions`` attribute
        or a plain list.

    Returns
    -------
    List[float]
        Prediction values.
    """
    if hasattr(result, "predictions"):
        return result.predictions  # type: ignore[union-attr]
    if isinstance(result, list):
        return result
    raise TypeError(
        f"Cannot extract predictions from {type(result)}"
    )


def _evaluate_with_cv(
    model: _Fittable,
    X: List[List[float]],
    y: List[float],
    cv_splits: List[Tuple[List[int], List[int]]],
    metric: Callable[[List[float], List[float]], float],
) -> float:
    """Evaluate model using pre-defined CV splits.

    Parameters
    ----------
    model : _Fittable
        Model implementing fit/predict.
    X : List[List[float]]
        Full feature matrix.
    y : List[float]
        Full target vector.
    cv_splits : List[Tuple[List[int], List[int]]]
        List of (train_indices, test_indices) tuples.
    metric : Callable
        Scoring function (y_true, y_pred) -> float. Higher is better.

    Returns
    -------
    float
        Mean score across all CV folds.
    """
    scores: List[float] = []
    for train_idx, test_idx in cv_splits:
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]

        model.fit(X_train, y_train)
        pred_result = model.predict(X_test)
        preds = _extract_predictions(pred_result)
        scores.append(metric(y_test, preds))

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _permute_column(
    X: List[List[float]], col: int, rng: random.Random
) -> List[List[float]]:
    """Create a copy of X with one column randomly permuted.

    Parameters
    ----------
    X : List[List[float]]
        Feature matrix.
    col : int
        Column index to permute.
    rng : random.Random
        Random number generator.

    Returns
    -------
    List[List[float]]
        Copy of X with column ``col`` shuffled.
    """
    n = len(X)
    # Extract and shuffle the column
    col_vals = [X[i][col] for i in range(n)]
    rng.shuffle(col_vals)

    result: List[List[float]] = []
    for i in range(n):
        row = X[i][:]
        row[col] = col_vals[i]
        result.append(row)
    return result


def _std_dev(values: List[float]) -> float:
    """Compute sample standard deviation.

    Parameters
    ----------
    values : List[float]
        Data values.

    Returns
    -------
    float
        Sample standard deviation.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mu = sum(values) / n
    var = sum((v - mu) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def time_series_feature_importance(
    model: _Fittable,
    X: List[List[float]],
    y: List[float],
    feature_names: List[str],
    cv_splits: List[Tuple[List[int], List[int]]],
    metric: Callable[[List[float], List[float]], float],
    n_repeats: int = 5,
    random_state: int = 42,
) -> TimeSeriesImportanceResult:
    """Compute permutation-based feature importance for time series.

    For each feature, the column is randomly permuted (breaking
    temporal structure) and the model is re-evaluated using the
    provided walk-forward CV splits. Importance is measured as the
    mean score drop relative to the baseline.

    Parameters
    ----------
    model : _Fittable
        Model implementing ``fit`` and ``predict``.
    X : List[List[float]]
        Feature matrix (n_samples x n_features).
    y : List[float]
        Target values (n_samples,).
    feature_names : List[str]
        Names for each feature column.
    cv_splits : List[Tuple[List[int], List[int]]]
        Walk-forward CV splits as (train_indices, test_indices).
    metric : Callable
        Scoring function (y_true, y_pred) -> float. Higher is
        better (e.g., negative MSE).
    n_repeats : int
        Number of permutation repeats per feature.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    TimeSeriesImportanceResult
        Feature importances sorted by importance (descending).

    Raises
    ------
    ValueError
        If feature_names length does not match number of columns.
    """
    if not X:
        raise ValueError("X must not be empty")
    n_features = len(X[0])
    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) must match "
            f"number of features ({n_features})"
        )

    rng = random.Random(random_state)

    # Compute baseline score
    baseline_score = _evaluate_with_cv(
        model, X, y, cv_splits, metric
    )

    # Compute importance for each feature
    raw_importances: List[List[float]] = []
    for j in range(n_features):
        drops: List[float] = []
        for _ in range(n_repeats):
            X_perm = _permute_column(X, j, rng)
            perm_score = _evaluate_with_cv(
                model, X_perm, y, cv_splits, metric
            )
            drops.append(baseline_score - perm_score)
        raw_importances.append(drops)

    # Aggregate
    mean_importances = [
        sum(drops) / len(drops) for drops in raw_importances
    ]
    std_importances = [
        _std_dev(drops) for drops in raw_importances
    ]

    # Sort by importance descending
    indices = sorted(
        range(n_features),
        key=lambda i: mean_importances[i],
        reverse=True,
    )

    return TimeSeriesImportanceResult(
        feature_names=[feature_names[i] for i in indices],
        importances=[mean_importances[i] for i in indices],
        importances_std=[std_importances[i] for i in indices],
        baseline_score=baseline_score,
    )


def sequential_feature_selection(
    model: _Fittable,
    X: List[List[float]],
    y: List[float],
    feature_names: List[str],
    cv_splits: List[Tuple[List[int], List[int]]],
    metric: Callable[[List[float], List[float]], float],
    direction: str = "forward",
    max_features: Optional[int] = None,
) -> List[str]:
    """Sequential feature selection for time series models.

    Greedily adds (forward) or removes (backward) features to
    maximise the metric evaluated via walk-forward CV.

    Parameters
    ----------
    model : _Fittable
        Model implementing ``fit`` and ``predict``.
    X : List[List[float]]
        Feature matrix (n_samples x n_features).
    y : List[float]
        Target values (n_samples,).
    feature_names : List[str]
        Names for each feature column.
    cv_splits : List[Tuple[List[int], List[int]]]
        Walk-forward CV splits as (train_indices, test_indices).
    metric : Callable
        Scoring function (y_true, y_pred) -> float. Higher is better.
    direction : str
        ``"forward"`` to add features or ``"backward"`` to remove.
    max_features : Optional[int]
        Maximum number of features to select (forward) or minimum
        to retain (backward). Defaults to all features.

    Returns
    -------
    List[str]
        Selected feature names in order of selection (forward)
        or order of removal (backward).

    Raises
    ------
    ValueError
        If direction is not ``"forward"`` or ``"backward"``.
    """
    if direction not in ("forward", "backward"):
        raise ValueError(
            f"direction must be 'forward' or 'backward', "
            f"got '{direction}'"
        )

    n_features = len(feature_names)
    if max_features is None:
        max_features = n_features

    def _subset_X(
        X_full: List[List[float]], cols: List[int]
    ) -> List[List[float]]:
        return [[row[c] for c in cols] for row in X_full]

    if direction == "forward":
        selected: List[int] = []
        remaining = list(range(n_features))
        selected_names: List[str] = []

        for _ in range(min(max_features, n_features)):
            best_score = -math.inf
            best_feat = -1

            for feat in remaining:
                candidate = selected + [feat]
                X_sub = _subset_X(X, candidate)
                score = _evaluate_with_cv(
                    model, X_sub, y, cv_splits, metric
                )
                if score > best_score:
                    best_score = score
                    best_feat = feat

            if best_feat < 0:
                break
            selected.append(best_feat)
            remaining.remove(best_feat)
            selected_names.append(feature_names[best_feat])

        return selected_names

    else:  # backward
        current = list(range(n_features))
        removed_names: List[str] = []
        min_keep = max(1, n_features - max_features)

        while len(current) > min_keep:
            best_score = -math.inf
            worst_feat = -1

            for feat in current:
                candidate = [f for f in current if f != feat]
                if not candidate:
                    break
                X_sub = _subset_X(X, candidate)
                score = _evaluate_with_cv(
                    model, X_sub, y, cv_splits, metric
                )
                if score > best_score:
                    best_score = score
                    worst_feat = feat

            if worst_feat < 0:
                break
            current.remove(worst_feat)
            removed_names.append(feature_names[worst_feat])

        # Return names of features that remain
        return [feature_names[i] for i in current]
