"""
Outlier Detection and Treatment

Detects and treats outliers in financial data using robust statistical
methods:

- IQR: Interquartile range fence method (Tukey's fences)
- MAD: Median absolute deviation (highly robust)
- ZSCORE: Standard deviation from mean

These methods are essential for financial data preprocessing where
fat-tailed distributions and market dislocations create extreme values
that can distort model estimation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from finceptquantlib.ml._matrix_utils import (
    column_mean,
    column_median,
    column_percentile,
    column_std,
)


class OutlierMethod(Enum):
    """Statistical method for outlier detection.

    Attributes
    ----------
    IQR
        Tukey's fences: outlier if x < Q1 - k*IQR or x > Q3 + k*IQR.
    MAD
        Median absolute deviation: outlier if |x - median| > k * MAD.
    ZSCORE
        Z-score: outlier if |x - mean| / std > k.
    """

    IQR = "iqr"
    MAD = "mad"
    ZSCORE = "zscore"


@dataclass
class OutlierResult:
    """Result of outlier detection.

    Attributes
    ----------
    outlier_mask : List[List[bool]]
        Boolean mask where True indicates an outlier
        (n_samples x n_features).
    n_outliers : int
        Total number of outlier values detected.
    bounds : Dict[int, Tuple[float, float]]
        Per-feature (lower, upper) bounds used for detection.
        Keys are column indices.
    """

    outlier_mask: List[List[bool]]
    n_outliers: int
    bounds: Dict[int, Tuple[float, float]] = field(
        default_factory=dict
    )


class OutlierDetector:
    """Detect and treat outliers in tabular data.

    Learns detection bounds from training data during ``fit()`` and
    applies them via ``detect()`` or ``treat()``.

    Parameters
    ----------
    method : str
        Detection method: ``"iqr"``, ``"mad"``, or ``"zscore"``.
        Default is ``"iqr"``.
    threshold : Optional[float]
        Sensitivity parameter. If None, uses method-specific defaults:
        1.5 for IQR, 3.0 for ZSCORE, 3.5 for MAD.

    Attributes
    ----------
    lower_bounds_ : Optional[List[float]]
        Per-feature lower detection bound learned during fit.
    upper_bounds_ : Optional[List[float]]
        Per-feature upper detection bound learned during fit.

    Examples
    --------
    >>> detector = OutlierDetector(method="iqr", threshold=1.5)
    >>> X = [[1.0], [2.0], [3.0], [4.0], [100.0]]
    >>> detector.fit(X)
    >>> result = detector.detect(X)
    >>> treated = detector.treat(X, method="clip")
    """

    _DEFAULT_THRESHOLDS = {
        OutlierMethod.IQR: 1.5,
        OutlierMethod.ZSCORE: 3.0,
        OutlierMethod.MAD: 3.5,
    }

    def __init__(
        self,
        method: str = "iqr",
        threshold: Optional[float] = None,
    ) -> None:
        try:
            self.method = OutlierMethod(method)
        except ValueError:
            valid = [m.value for m in OutlierMethod]
            raise ValueError(
                f"Unknown method '{method}'. Expected one of {valid}."
            ) from None
        if threshold is not None:
            self.threshold = threshold
        else:
            self.threshold = self._DEFAULT_THRESHOLDS[self.method]
        self.lower_bounds_: Optional[List[float]] = None
        self.upper_bounds_: Optional[List[float]] = None

    def fit(self, X: List[List[float]]) -> None:
        """Learn outlier detection bounds from training data.

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

        if self.method == OutlierMethod.IQR:
            self._fit_iqr(X)
        elif self.method == OutlierMethod.MAD:
            self._fit_mad(X)
        elif self.method == OutlierMethod.ZSCORE:
            self._fit_zscore(X)

    def _fit_iqr(self, X: List[List[float]]) -> None:
        """Compute IQR-based bounds (Tukey's fences).

        Parameters
        ----------
        X : List[List[float]]
            Training data.
        """
        q25 = column_percentile(X, 0.25)
        q75 = column_percentile(X, 0.75)
        m = len(q25)
        k = self.threshold
        self.lower_bounds_ = [
            q25[j] - k * (q75[j] - q25[j]) for j in range(m)
        ]
        self.upper_bounds_ = [
            q75[j] + k * (q75[j] - q25[j]) for j in range(m)
        ]

    def _fit_mad(self, X: List[List[float]]) -> None:
        """Compute MAD-based bounds.

        The MAD is scaled by 1.4826 to be a consistent estimator
        of the standard deviation for normally distributed data.

        Parameters
        ----------
        X : List[List[float]]
            Training data.
        """
        medians = column_median(X)
        m = len(medians)
        n = len(X)
        # Compute MAD for each column
        mad_values: List[float] = []
        for j in range(m):
            abs_devs = sorted(
                abs(X[i][j] - medians[j]) for i in range(n)
            )
            if n % 2 == 1:
                mad_j = abs_devs[n // 2]
            else:
                mad_j = (
                    abs_devs[n // 2 - 1] + abs_devs[n // 2]
                ) / 2.0
            mad_values.append(mad_j)

        # Scale factor for consistency with normal distribution
        consistency_factor = 1.4826
        k = self.threshold
        self.lower_bounds_ = [
            medians[j] - k * consistency_factor * mad_values[j]
            for j in range(m)
        ]
        self.upper_bounds_ = [
            medians[j] + k * consistency_factor * mad_values[j]
            for j in range(m)
        ]

    def _fit_zscore(self, X: List[List[float]]) -> None:
        """Compute z-score-based bounds.

        Parameters
        ----------
        X : List[List[float]]
            Training data.
        """
        means = column_mean(X)
        stds = column_std(X, ddof=0)
        m = len(means)
        k = self.threshold
        self.lower_bounds_ = [
            means[j] - k * stds[j] for j in range(m)
        ]
        self.upper_bounds_ = [
            means[j] + k * stds[j] for j in range(m)
        ]

    def detect(self, X: List[List[float]]) -> OutlierResult:
        """Detect outliers in data using learned bounds.

        Parameters
        ----------
        X : List[List[float]]
            Data to check for outliers (n_samples x n_features).

        Returns
        -------
        OutlierResult
            Detection result with boolean mask, count, and bounds.

        Raises
        ------
        RuntimeError
            If detect is called before fit.
        """
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise RuntimeError("Must call fit() before detect().")
        m = len(self.lower_bounds_)
        mask: List[List[bool]] = []
        total_outliers = 0
        for row in X:
            row_mask: List[bool] = []
            for j in range(m):
                is_outlier = (
                    row[j] < self.lower_bounds_[j]
                    or row[j] > self.upper_bounds_[j]
                )
                row_mask.append(is_outlier)
                if is_outlier:
                    total_outliers += 1
            mask.append(row_mask)

        bounds: Dict[int, Tuple[float, float]] = {
            j: (self.lower_bounds_[j], self.upper_bounds_[j])
            for j in range(m)
        }
        return OutlierResult(
            outlier_mask=mask,
            n_outliers=total_outliers,
            bounds=bounds,
        )

    def treat(
        self,
        X: List[List[float]],
        method: str = "clip",
    ) -> List[List[float]]:
        """Treat outliers by clipping or replacing with bounds.

        Parameters
        ----------
        X : List[List[float]]
            Data to treat (n_samples x n_features).
        method : str
            Treatment method. ``"clip"`` clips outliers to the
            detection bounds. ``"nan"`` replaces outliers with
            ``float('nan')``. Default is ``"clip"``.

        Returns
        -------
        List[List[float]]
            Treated data with outliers handled.

        Raises
        ------
        RuntimeError
            If treat is called before fit.
        ValueError
            If treatment method is unknown.
        """
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise RuntimeError("Must call fit() before treat().")
        if method not in ("clip", "nan"):
            raise ValueError(
                f"Unknown treatment method '{method}'. "
                f"Expected 'clip' or 'nan'."
            )
        m = len(self.lower_bounds_)
        result: List[List[float]] = []
        for row in X:
            new_row: List[float] = []
            for j in range(m):
                val = row[j]
                is_outlier = (
                    val < self.lower_bounds_[j]
                    or val > self.upper_bounds_[j]
                )
                if is_outlier:
                    if method == "clip":
                        val = max(
                            self.lower_bounds_[j],
                            min(val, self.upper_bounds_[j]),
                        )
                    else:  # nan
                        val = float("nan")
                new_row.append(val)
            result.append(new_row)
        return result
