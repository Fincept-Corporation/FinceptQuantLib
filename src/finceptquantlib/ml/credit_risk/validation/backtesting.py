"""
ML Model Backtesting Framework

Provides cross-validated backtesting for ML models, computing
out-of-sample performance metrics across multiple folds. Supports
any model implementing fit/predict and any cross-validation splitter
implementing split(n_samples).

All operations use pure Python lists without external dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, List, Tuple


@dataclass
class BacktestResult:
    """Result of a cross-validated backtest run.

    Attributes
    ----------
    metric_values : List[float]
        Metric score for each fold.
    mean_metric : float
        Mean of metric values across folds.
    std_metric : float
        Standard deviation of metric values across folds.
    predictions : List[float]
        Concatenated out-of-sample predictions across all folds.
    actuals : List[float]
        Concatenated actual values corresponding to predictions.
    fold_sizes : List[Tuple[int, int]]
        (train_size, test_size) for each fold.
    n_folds : int
        Number of cross-validation folds.
    """

    metric_values: List[float]
    mean_metric: float
    std_metric: float
    predictions: List[float]
    actuals: List[float]
    fold_sizes: List[Tuple[int, int]]
    n_folds: int


class MLBacktester:
    """Cross-validated backtester for ML models.

    Runs a model through cross-validation folds, collecting
    out-of-sample predictions and computing a user-specified
    evaluation metric on each fold.

    Parameters
    ----------
    model : Any
        Model with ``fit(X, y)`` and ``predict(X)`` methods.
        The ``predict`` method must return an object with a
        ``predictions`` attribute (e.g., ``PredictResult``).
    cv : Any
        Cross-validation splitter with a ``split(n_samples)`` method
        returning ``List[Tuple[List[int], List[int]]]`` of
        (train_indices, test_indices) pairs.
    metric : Callable[[List[float], List[float]], float]
        Evaluation function taking (y_true, y_pred) and returning
        a scalar score.

    Examples
    --------
    >>> from finceptquantlib.ml.credit_risk.validation.backtesting import MLBacktester
    >>> backtester = MLBacktester(model=my_model, cv=my_cv, metric=mse)  # doctest: +SKIP
    >>> result = backtester.run(X_train, y_train)  # doctest: +SKIP
    >>> print(f"Mean MSE: {result.mean_metric:.4f}")  # doctest: +SKIP
    """

    def __init__(
        self,
        model: Any,
        cv: Any,
        metric: Callable[[List[float], List[float]], float],
    ) -> None:
        self._model = model
        self._cv = cv
        self._metric = metric

    def run(
        self,
        X: List[List[float]],
        y: List[float],
    ) -> BacktestResult:
        """Execute cross-validated backtest.

        For each fold produced by the cross-validation splitter:
        1. Extract training and test subsets using fold indices.
        2. Fit the model on the training subset.
        3. Generate predictions on the test subset.
        4. Evaluate the metric on test actuals vs predictions.

        Parameters
        ----------
        X : List[List[float]]
            Feature matrix (n_samples x n_features).
        y : List[float]
            Target values (n_samples,).

        Returns
        -------
        BacktestResult
            Aggregated results across all folds.

        Raises
        ------
        ValueError
            If X and y have mismatched lengths or are empty.
        """
        n_samples = len(X)
        if n_samples == 0:
            raise ValueError("X must not be empty")
        if len(y) != n_samples:
            raise ValueError(
                f"X has {n_samples} samples but y has {len(y)}"
            )

        splits = self._cv.split(n_samples)

        metric_values: List[float] = []
        all_predictions: List[float] = []
        all_actuals: List[float] = []
        fold_sizes: List[Tuple[int, int]] = []

        for train_idx, test_idx in splits:
            # Extract training data
            X_train = [X[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]

            # Extract test data
            X_test = [X[i] for i in test_idx]
            y_test = [y[i] for i in test_idx]

            # Fit and predict
            self._model.fit(X_train, y_train)
            pred_result = self._model.predict(X_test)
            predictions = pred_result.predictions

            # Evaluate
            score = self._metric(y_test, predictions)
            metric_values.append(score)

            # Collect results
            all_predictions.extend(predictions)
            all_actuals.extend(y_test)
            fold_sizes.append((len(train_idx), len(test_idx)))

        n_folds = len(metric_values)
        mean_metric = sum(metric_values) / n_folds if n_folds > 0 else 0.0
        if n_folds > 1:
            variance = sum(
                (v - mean_metric) ** 2 for v in metric_values
            ) / (n_folds - 1)
            std_metric = math.sqrt(variance)
        else:
            std_metric = 0.0

        return BacktestResult(
            metric_values=metric_values,
            mean_metric=mean_metric,
            std_metric=std_metric,
            predictions=all_predictions,
            actuals=all_actuals,
            fold_sizes=fold_sizes,
            n_folds=n_folds,
        )
