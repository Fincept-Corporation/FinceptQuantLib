"""
Time Series Cross-Validation Strategies

Implements walk-forward, expanding window, and purged K-fold
cross-validation schemes for temporal data. These prevent
look-ahead bias that standard random CV introduces in financial
time series.

All computations are pure Python.
"""

from __future__ import annotations

import math
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
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CVResult:
    """Result of cross-validation evaluation.

    Attributes
    ----------
    scores : List[float]
        Score for each fold.
    mean_score : float
        Mean score across folds.
    std_score : float
        Standard deviation of scores across folds.
    train_sizes : List[int]
        Training set size for each fold.
    test_sizes : List[int]
        Test set size for each fold.
    fold_predictions : Optional[List[List[float]]]
        Predictions for each fold's test set. ``None`` if not
        requested.
    """

    scores: List[float]
    mean_score: float
    std_score: float
    train_sizes: List[int]
    test_sizes: List[int]
    fold_predictions: Optional[List[List[float]]] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_predictions(result: object) -> List[float]:
    """Extract prediction list from a model result object.

    Parameters
    ----------
    result : object
        Either a PredictResult with ``.predictions`` or a list.

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


def _std_dev(values: List[float]) -> float:
    """Compute sample standard deviation.

    Parameters
    ----------
    values : List[float]
        Values to compute std for.

    Returns
    -------
    float
        Sample standard deviation (ddof=1).
    """
    n = len(values)
    if n < 2:
        return 0.0
    mu = sum(values) / n
    var = sum((v - mu) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)


# ---------------------------------------------------------------------------
# WalkForwardCV
# ---------------------------------------------------------------------------

class WalkForwardCV:
    """Walk-forward (rolling window) cross-validation.

    Generates time-respecting train/test splits by sliding a fixed-
    or expanding-size training window forward through the data.

    Parameters
    ----------
    n_splits : int
        Number of CV folds to generate.
    train_size : Optional[int]
        Fixed training window size. If ``None``, uses an expanding
        window starting from the first observation.
    test_size : int
        Number of test observations per fold.
    gap : int
        Number of observations to skip between training end and
        test start to avoid information leakage.

    Examples
    --------
    >>> cv = WalkForwardCV(n_splits=3, train_size=50, test_size=10)
    >>> splits = cv.split(100)
    >>> len(splits)
    3
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_size: Optional[int] = None,
        test_size: int = 1,
        gap: int = 0,
    ) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if test_size < 1:
            raise ValueError("test_size must be >= 1")
        if gap < 0:
            raise ValueError("gap must be >= 0")
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap

    def split(
        self, n_samples: int
    ) -> List[Tuple[List[int], List[int]]]:
        """Generate train/test index splits.

        Parameters
        ----------
        n_samples : int
            Total number of samples in the dataset.

        Returns
        -------
        List[Tuple[List[int], List[int]]]
            List of (train_indices, test_indices) tuples.

        Raises
        ------
        ValueError
            If n_samples is too small for the requested splits.
        """
        splits: List[Tuple[List[int], List[int]]] = []

        if self.train_size is not None:
            # Fixed window: slide forward
            # First test starts after train_size + gap
            first_test_start = self.train_size + self.gap
            # Step between consecutive test starts
            step = max(
                1,
                (n_samples - first_test_start - self.test_size)
                // max(self.n_splits - 1, 1)
            ) if self.n_splits > 1 else self.test_size

            for i in range(self.n_splits):
                test_start = first_test_start + i * step
                test_end = min(test_start + self.test_size, n_samples)
                if test_start >= n_samples:
                    break
                train_start = test_start - self.gap - self.train_size
                train_start = max(0, train_start)
                train_end = test_start - self.gap

                if train_end <= train_start:
                    continue

                train_idx = list(range(train_start, train_end))
                test_idx = list(range(test_start, test_end))
                if train_idx and test_idx:
                    splits.append((train_idx, test_idx))
        else:
            # Expanding window
            # Total usable = n_samples - gap
            # We need at least 1 training sample per fold
            min_train = max(1, (n_samples - self.gap) // (self.n_splits + 1))
            step = max(
                self.test_size,
                (n_samples - min_train - self.gap)
                // max(self.n_splits, 1),
            )

            for i in range(self.n_splits):
                test_start = min_train + self.gap + i * step
                test_end = min(
                    test_start + self.test_size, n_samples
                )
                if test_start >= n_samples:
                    break
                train_end = test_start - self.gap
                if train_end <= 0:
                    continue

                train_idx = list(range(0, train_end))
                test_idx = list(range(test_start, test_end))
                if train_idx and test_idx:
                    splits.append((train_idx, test_idx))

        return splits

    def evaluate(
        self,
        model: _Fittable,
        X: List[List[float]],
        y: List[float],
        metric: Callable[[List[float], List[float]], float],
    ) -> CVResult:
        """Evaluate a model using walk-forward CV.

        Parameters
        ----------
        model : _Fittable
            Model implementing ``fit`` and ``predict``.
        X : List[List[float]]
            Feature matrix (n_samples x n_features).
        y : List[float]
            Target values (n_samples,).
        metric : Callable
            Scoring function (y_true, y_pred) -> float.

        Returns
        -------
        CVResult
            Cross-validation results with per-fold scores.
        """
        n_samples = len(X)
        splits = self.split(n_samples)
        return _run_cv(model, X, y, splits, metric)


# ---------------------------------------------------------------------------
# ExpandingWindowCV
# ---------------------------------------------------------------------------

class ExpandingWindowCV:
    """Expanding window cross-validation.

    Training window starts at ``initial_train_size`` and grows by
    ``test_size`` each fold. Optionally caps training window at
    ``max_train_size``.

    Parameters
    ----------
    initial_train_size : int
        Minimum training window size for the first fold.
    test_size : int
        Number of test observations per fold.
    gap : int
        Gap between training end and test start.
    max_train_size : Optional[int]
        Maximum training window size. If set, older observations
        are dropped once this limit is reached.

    Examples
    --------
    >>> cv = ExpandingWindowCV(initial_train_size=30, test_size=5)
    >>> splits = cv.split(100)
    >>> len(splits) > 0
    True
    """

    def __init__(
        self,
        initial_train_size: int = 50,
        test_size: int = 1,
        gap: int = 0,
        max_train_size: Optional[int] = None,
    ) -> None:
        if initial_train_size < 1:
            raise ValueError("initial_train_size must be >= 1")
        if test_size < 1:
            raise ValueError("test_size must be >= 1")
        if gap < 0:
            raise ValueError("gap must be >= 0")
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.gap = gap
        self.max_train_size = max_train_size

    def split(
        self, n_samples: int
    ) -> List[Tuple[List[int], List[int]]]:
        """Generate expanding window train/test splits.

        Parameters
        ----------
        n_samples : int
            Total number of samples.

        Returns
        -------
        List[Tuple[List[int], List[int]]]
            List of (train_indices, test_indices) tuples.
        """
        splits: List[Tuple[List[int], List[int]]] = []

        test_start = self.initial_train_size + self.gap
        while test_start < n_samples:
            test_end = min(
                test_start + self.test_size, n_samples
            )
            train_end = test_start - self.gap

            if self.max_train_size is not None:
                train_start = max(
                    0, train_end - self.max_train_size
                )
            else:
                train_start = 0

            if train_end <= train_start:
                test_start += self.test_size
                continue

            train_idx = list(range(train_start, train_end))
            test_idx = list(range(test_start, test_end))
            if train_idx and test_idx:
                splits.append((train_idx, test_idx))

            test_start += self.test_size

        return splits

    def evaluate(
        self,
        model: _Fittable,
        X: List[List[float]],
        y: List[float],
        metric: Callable[[List[float], List[float]], float],
    ) -> CVResult:
        """Evaluate a model using expanding window CV.

        Parameters
        ----------
        model : _Fittable
            Model implementing ``fit`` and ``predict``.
        X : List[List[float]]
            Feature matrix (n_samples x n_features).
        y : List[float]
            Target values (n_samples,).
        metric : Callable
            Scoring function (y_true, y_pred) -> float.

        Returns
        -------
        CVResult
            Cross-validation results.
        """
        n_samples = len(X)
        splits = self.split(n_samples)
        return _run_cv(model, X, y, splits, metric)


# ---------------------------------------------------------------------------
# PurgedKFoldCV
# ---------------------------------------------------------------------------

class PurgedKFoldCV:
    """Purged K-Fold cross-validation.

    Based on Marcos Lopez de Prado's methodology. Standard K-fold
    splits are generated chronologically, then observations within
    ``purge_gap`` of each test fold boundary are removed from the
    training set to prevent information leakage from overlapping
    financial features (e.g., rolling averages, labelling windows).

    Parameters
    ----------
    n_splits : int
        Number of folds.
    purge_gap : int
        Number of samples to purge before and after each test fold.
        Observations in ``[test_start - purge_gap, test_start)``
        and ``(test_end, test_end + purge_gap]`` are removed from
        training.

    Examples
    --------
    >>> cv = PurgedKFoldCV(n_splits=5, purge_gap=3)
    >>> splits = cv.split(100)
    >>> len(splits) == 5
    True
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 5,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if purge_gap < 0:
            raise ValueError("purge_gap must be >= 0")
        self.n_splits = n_splits
        self.purge_gap = purge_gap

    def split(
        self, n_samples: int
    ) -> List[Tuple[List[int], List[int]]]:
        """Generate purged K-fold train/test splits.

        Samples are divided into ``n_splits`` contiguous folds in
        chronological order. For each fold used as the test set,
        ``purge_gap`` observations before and after the test fold
        boundaries are removed from the training indices.

        Parameters
        ----------
        n_samples : int
            Total number of samples.

        Returns
        -------
        List[Tuple[List[int], List[int]]]
            List of (train_indices, test_indices) tuples.

        Raises
        ------
        ValueError
            If n_samples < n_splits.
        """
        if n_samples < self.n_splits:
            raise ValueError(
                f"n_samples ({n_samples}) must be >= "
                f"n_splits ({self.n_splits})"
            )

        # Create contiguous fold boundaries
        fold_size = n_samples // self.n_splits
        remainder = n_samples % self.n_splits

        fold_starts: List[int] = []
        fold_ends: List[int] = []
        start = 0
        for i in range(self.n_splits):
            end = start + fold_size + (1 if i < remainder else 0)
            fold_starts.append(start)
            fold_ends.append(end)
            start = end

        splits: List[Tuple[List[int], List[int]]] = []

        for k in range(self.n_splits):
            test_start = fold_starts[k]
            test_end = fold_ends[k]
            test_idx = list(range(test_start, test_end))

            # Purge zone: remove observations near test boundaries
            purge_start = max(0, test_start - self.purge_gap)
            purge_end = min(n_samples, test_end + self.purge_gap)
            purge_set = set(range(purge_start, purge_end))

            train_idx = [
                i for i in range(n_samples)
                if i not in purge_set
                and (i < test_start or i >= test_end)
            ]

            if train_idx and test_idx:
                splits.append((train_idx, test_idx))

        return splits

    def evaluate(
        self,
        model: _Fittable,
        X: List[List[float]],
        y: List[float],
        metric: Callable[[List[float], List[float]], float],
    ) -> CVResult:
        """Evaluate a model using purged K-fold CV.

        Parameters
        ----------
        model : _Fittable
            Model implementing ``fit`` and ``predict``.
        X : List[List[float]]
            Feature matrix (n_samples x n_features).
        y : List[float]
            Target values (n_samples,).
        metric : Callable
            Scoring function (y_true, y_pred) -> float.

        Returns
        -------
        CVResult
            Cross-validation results.
        """
        n_samples = len(X)
        splits = self.split(n_samples)
        return _run_cv(model, X, y, splits, metric)


# ---------------------------------------------------------------------------
# Shared evaluation runner
# ---------------------------------------------------------------------------

def _run_cv(
    model: _Fittable,
    X: List[List[float]],
    y: List[float],
    splits: List[Tuple[List[int], List[int]]],
    metric: Callable[[List[float], List[float]], float],
) -> CVResult:
    """Run cross-validation over pre-computed splits.

    Parameters
    ----------
    model : _Fittable
        Model implementing ``fit`` and ``predict``.
    X : List[List[float]]
        Feature matrix.
    y : List[float]
        Target values.
    splits : List[Tuple[List[int], List[int]]]
        Pre-computed (train, test) index tuples.
    metric : Callable
        Scoring function.

    Returns
    -------
    CVResult
        Aggregated cross-validation results.
    """
    scores: List[float] = []
    train_sizes: List[int] = []
    test_sizes: List[int] = []
    fold_preds: List[List[float]] = []

    for train_idx, test_idx in splits:
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]

        model.fit(X_train, y_train)
        pred_result = model.predict(X_test)
        preds = _extract_predictions(pred_result)

        scores.append(metric(y_test, preds))
        train_sizes.append(len(train_idx))
        test_sizes.append(len(test_idx))
        fold_preds.append(preds)

    mean_score = sum(scores) / len(scores) if scores else 0.0
    std_score = _std_dev(scores)

    return CVResult(
        scores=scores,
        mean_score=mean_score,
        std_score=std_score,
        train_sizes=train_sizes,
        test_sizes=test_sizes,
        fold_predictions=fold_preds,
    )
