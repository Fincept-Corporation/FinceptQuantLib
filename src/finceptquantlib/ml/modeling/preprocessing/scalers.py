"""
Feature Scaling Transformers

Provides standardization and normalization transformers for financial
time-series and cross-sectional data:

- ZScoreScaler: Standardize to zero mean and unit variance
- MinMaxScaler: Scale features to a specified range
- RobustScaler: Scale using median and interquartile range (outlier-robust)

All transformers implement the Transformer ABC from finceptquantlib.ml._base.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from finceptquantlib.ml._base import Transformer
from finceptquantlib.ml._matrix_utils import (
    column_mean,
    column_std,
    column_min,
    column_max,
    column_median,
    column_percentile,
)


class ZScoreScaler(Transformer):
    """Standardize features by removing mean and scaling to unit variance.

    Each feature is transformed as::

        z = (x - mean) / std

    Features with zero standard deviation are left unchanged (not scaled).

    Attributes
    ----------
    mean_ : Optional[List[float]]
        Per-feature mean learned during fit. None before fit.
    std_ : Optional[List[float]]
        Per-feature standard deviation learned during fit. None before fit.

    Examples
    --------
    >>> scaler = ZScoreScaler()
    >>> X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    >>> scaler.fit(X)
    >>> transformed = scaler.transform(X)
    >>> original = scaler.inverse_transform(transformed)
    """

    def __init__(self) -> None:
        self.mean_: Optional[List[float]] = None
        self.std_: Optional[List[float]] = None

    def fit(self, X: List[List[float]]) -> None:
        """Learn mean and standard deviation from training data.

        Parameters
        ----------
        X : List[List[float]]
            Training data (n_samples x n_features).

        Raises
        ------
        ValueError
            If X is empty.
        """
        if not X:
            raise ValueError("Cannot fit on empty data.")
        self.mean_ = column_mean(X)
        self.std_ = column_std(X, ddof=0)

    def transform(self, X: List[List[float]]) -> List[List[float]]:
        """Standardize features using learned mean and std.

        Parameters
        ----------
        X : List[List[float]]
            Data to transform (n_samples x n_features).

        Returns
        -------
        List[List[float]]
            Standardized data.

        Raises
        ------
        RuntimeError
            If transform is called before fit.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Must call fit() before transform().")
        m = len(self.mean_)
        result: List[List[float]] = []
        for row in X:
            new_row: List[float] = []
            for j in range(m):
                if self.std_[j] > 0.0:
                    new_row.append((row[j] - self.mean_[j]) / self.std_[j])
                else:
                    new_row.append(row[j])
            result.append(new_row)
        return result

    def inverse_transform(
        self, X: List[List[float]]
    ) -> List[List[float]]:
        """Reverse standardization to recover original scale.

        Parameters
        ----------
        X : List[List[float]]
            Standardized data.

        Returns
        -------
        List[List[float]]
            Data in original scale.

        Raises
        ------
        RuntimeError
            If inverse_transform is called before fit.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Must call fit() before inverse_transform().")
        m = len(self.mean_)
        result: List[List[float]] = []
        for row in X:
            new_row: List[float] = []
            for j in range(m):
                if self.std_[j] > 0.0:
                    new_row.append(row[j] * self.std_[j] + self.mean_[j])
                else:
                    new_row.append(row[j])
            result.append(new_row)
        return result


class MinMaxScaler(Transformer):
    """Scale features to a specified range using min-max normalization.

    Each feature is transformed as::

        x_scaled = (x - min) / (max - min) * (range_max - range_min)
                   + range_min

    Parameters
    ----------
    feature_range : Tuple[float, float]
        Desired range of transformed data. Default is (0.0, 1.0).

    Attributes
    ----------
    min_ : Optional[List[float]]
        Per-feature minimum learned during fit.
    max_ : Optional[List[float]]
        Per-feature maximum learned during fit.
    feature_range : Tuple[float, float]
        Target output range.

    Examples
    --------
    >>> scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    >>> X = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
    >>> scaler.fit(X)
    >>> scaler.transform([[2.0, 20.0]])
    [[0.5, 0.5]]
    """

    def __init__(
        self, feature_range: Tuple[float, float] = (0.0, 1.0)
    ) -> None:
        self.feature_range = feature_range
        self.min_: Optional[List[float]] = None
        self.max_: Optional[List[float]] = None

    def fit(self, X: List[List[float]]) -> None:
        """Learn per-feature minimum and maximum from training data.

        Parameters
        ----------
        X : List[List[float]]
            Training data (n_samples x n_features).

        Raises
        ------
        ValueError
            If X is empty or feature_range is invalid.
        """
        if not X:
            raise ValueError("Cannot fit on empty data.")
        if self.feature_range[0] >= self.feature_range[1]:
            raise ValueError(
                "feature_range[0] must be strictly less than "
                "feature_range[1]."
            )
        self.min_ = column_min(X)
        self.max_ = column_max(X)

    def transform(self, X: List[List[float]]) -> List[List[float]]:
        """Scale features to the target range.

        Parameters
        ----------
        X : List[List[float]]
            Data to transform (n_samples x n_features).

        Returns
        -------
        List[List[float]]
            Scaled data within feature_range.

        Raises
        ------
        RuntimeError
            If transform is called before fit.
        """
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("Must call fit() before transform().")
        m = len(self.min_)
        r_min, r_max = self.feature_range
        r_span = r_max - r_min
        result: List[List[float]] = []
        for row in X:
            new_row: List[float] = []
            for j in range(m):
                data_range = self.max_[j] - self.min_[j]
                if data_range > 0.0:
                    scaled = (row[j] - self.min_[j]) / data_range
                    new_row.append(scaled * r_span + r_min)
                else:
                    new_row.append(r_min)
            result.append(new_row)
        return result

    def inverse_transform(
        self, X: List[List[float]]
    ) -> List[List[float]]:
        """Reverse min-max scaling to recover original scale.

        Parameters
        ----------
        X : List[List[float]]
            Scaled data.

        Returns
        -------
        List[List[float]]
            Data in original scale.

        Raises
        ------
        RuntimeError
            If inverse_transform is called before fit.
        """
        if self.min_ is None or self.max_ is None:
            raise RuntimeError(
                "Must call fit() before inverse_transform()."
            )
        m = len(self.min_)
        r_min, r_max = self.feature_range
        r_span = r_max - r_min
        result: List[List[float]] = []
        for row in X:
            new_row: List[float] = []
            for j in range(m):
                data_range = self.max_[j] - self.min_[j]
                if data_range > 0.0 and r_span > 0.0:
                    unscaled = (row[j] - r_min) / r_span
                    new_row.append(
                        unscaled * data_range + self.min_[j]
                    )
                else:
                    new_row.append(self.min_[j])
            result.append(new_row)
        return result


class RobustScaler(Transformer):
    """Scale features using statistics robust to outliers.

    Centers data by the median and scales by the interquartile range
    (IQR), making this scaler robust to outliers.

    Each feature is transformed as::

        x_scaled = (x - median) / IQR

    where IQR = Q_upper - Q_lower (by default, 75th - 25th percentile).

    Parameters
    ----------
    quantile_range : Tuple[float, float]
        Quantile range used to compute the IQR. Values must be in
        [0, 1]. Default is (0.25, 0.75).

    Attributes
    ----------
    median_ : Optional[List[float]]
        Per-feature median learned during fit.
    iqr_ : Optional[List[float]]
        Per-feature interquartile range learned during fit.

    Examples
    --------
    >>> scaler = RobustScaler(quantile_range=(0.25, 0.75))
    >>> X = [[1.0], [2.0], [3.0], [4.0], [100.0]]
    >>> scaler.fit(X)
    >>> scaler.transform([[3.0]])
    [[0.0]]
    """

    def __init__(
        self,
        quantile_range: Tuple[float, float] = (0.25, 0.75),
    ) -> None:
        self.quantile_range = quantile_range
        self.median_: Optional[List[float]] = None
        self.iqr_: Optional[List[float]] = None
        self._q_lower: Optional[List[float]] = None
        self._q_upper: Optional[List[float]] = None

    def fit(self, X: List[List[float]]) -> None:
        """Learn median and IQR from training data.

        Parameters
        ----------
        X : List[List[float]]
            Training data (n_samples x n_features).

        Raises
        ------
        ValueError
            If X is empty or quantile_range is invalid.
        """
        if not X:
            raise ValueError("Cannot fit on empty data.")
        q_lo, q_hi = self.quantile_range
        if not (0.0 <= q_lo < q_hi <= 1.0):
            raise ValueError(
                "quantile_range must satisfy 0 <= q_lo < q_hi <= 1."
            )
        self.median_ = column_median(X)
        self._q_lower = column_percentile(X, q_lo)
        self._q_upper = column_percentile(X, q_hi)
        self.iqr_ = [
            self._q_upper[j] - self._q_lower[j]
            for j in range(len(self.median_))
        ]

    def transform(self, X: List[List[float]]) -> List[List[float]]:
        """Scale features using median and IQR.

        Parameters
        ----------
        X : List[List[float]]
            Data to transform (n_samples x n_features).

        Returns
        -------
        List[List[float]]
            Robustly scaled data.

        Raises
        ------
        RuntimeError
            If transform is called before fit.
        """
        if self.median_ is None or self.iqr_ is None:
            raise RuntimeError("Must call fit() before transform().")
        m = len(self.median_)
        result: List[List[float]] = []
        for row in X:
            new_row: List[float] = []
            for j in range(m):
                if self.iqr_[j] > 0.0:
                    new_row.append(
                        (row[j] - self.median_[j]) / self.iqr_[j]
                    )
                else:
                    new_row.append(row[j] - self.median_[j])
            result.append(new_row)
        return result

    def inverse_transform(
        self, X: List[List[float]]
    ) -> List[List[float]]:
        """Reverse robust scaling to recover original scale.

        Parameters
        ----------
        X : List[List[float]]
            Scaled data.

        Returns
        -------
        List[List[float]]
            Data in original scale.

        Raises
        ------
        RuntimeError
            If inverse_transform is called before fit.
        """
        if self.median_ is None or self.iqr_ is None:
            raise RuntimeError(
                "Must call fit() before inverse_transform()."
            )
        m = len(self.median_)
        result: List[List[float]] = []
        for row in X:
            new_row: List[float] = []
            for j in range(m):
                if self.iqr_[j] > 0.0:
                    new_row.append(
                        row[j] * self.iqr_[j] + self.median_[j]
                    )
                else:
                    new_row.append(row[j] + self.median_[j])
            result.append(new_row)
        return result
