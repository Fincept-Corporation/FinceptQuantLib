"""
Winsorization Transformer

Clips extreme values to reduce the impact of outliers on downstream
models. Supports two methods:

- PERCENTILE: Clip at specified lower/upper percentiles
- IQR: Clip at median +/- k * IQR

Winsorization is widely used in quantitative finance to handle fat-tailed
return distributions before model fitting.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from finceptquantlib.ml._base import Transformer
from finceptquantlib.ml._matrix_utils import (
    column_median,
    column_percentile,
)


class WinsorizeMethod(Enum):
    """Method used to determine winsorization bounds.

    Attributes
    ----------
    PERCENTILE
        Clip at specified lower and upper percentiles.
    IQR
        Clip at median +/- factor * IQR.
    """

    PERCENTILE = "percentile"
    IQR = "iqr"


class Winsorizer(Transformer):
    """Clip extreme values to reduce outlier influence.

    Learns clipping bounds from training data during ``fit()`` and
    applies them during ``transform()``.

    Parameters
    ----------
    lower : float
        Lower bound parameter. For PERCENTILE method, this is the
        lower percentile (e.g., 0.01 for 1st percentile). For IQR
        method, this is the multiplier below the median.
        Default is 0.01.
    upper : float
        Upper bound parameter. For PERCENTILE method, this is the
        upper percentile (e.g., 0.99 for 99th percentile). For IQR
        method, this is the multiplier above the median.
        Default is 0.99.
    method : str
        Winsorization method: ``"percentile"`` or ``"iqr"``.
        Default is ``"percentile"``.

    Attributes
    ----------
    lower_bounds_ : Optional[List[float]]
        Per-feature lower clip bound learned during fit.
    upper_bounds_ : Optional[List[float]]
        Per-feature upper clip bound learned during fit.

    Examples
    --------
    >>> w = Winsorizer(lower=0.05, upper=0.95, method="percentile")
    >>> X = [[r] for r in range(100)]
    >>> w.fit(X)
    >>> w.transform([[-50.0], [150.0]])
    [[4.95], [94.05]]
    """

    def __init__(
        self,
        lower: float = 0.01,
        upper: float = 0.99,
        method: str = "percentile",
    ) -> None:
        self.lower = lower
        self.upper = upper
        try:
            self.method = WinsorizeMethod(method)
        except ValueError:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Expected 'percentile' or 'iqr'."
            ) from None
        self.lower_bounds_: Optional[List[float]] = None
        self.upper_bounds_: Optional[List[float]] = None

    def fit(self, X: List[List[float]]) -> None:
        """Learn clipping bounds from training data.

        Parameters
        ----------
        X : List[List[float]]
            Training data (n_samples x n_features).

        Raises
        ------
        ValueError
            If X is empty or parameters are invalid.
        """
        if not X:
            raise ValueError("Cannot fit on empty data.")

        if self.method == WinsorizeMethod.PERCENTILE:
            if not (0.0 <= self.lower < self.upper <= 1.0):
                raise ValueError(
                    "For percentile method, require "
                    "0 <= lower < upper <= 1."
                )
            self.lower_bounds_ = column_percentile(X, self.lower)
            self.upper_bounds_ = column_percentile(X, self.upper)

        elif self.method == WinsorizeMethod.IQR:
            if self.lower < 0.0 or self.upper < 0.0:
                raise ValueError(
                    "For IQR method, lower and upper multipliers "
                    "must be non-negative."
                )
            medians = column_median(X)
            q25 = column_percentile(X, 0.25)
            q75 = column_percentile(X, 0.75)
            m = len(medians)
            self.lower_bounds_ = [
                medians[j] - self.lower * (q75[j] - q25[j])
                for j in range(m)
            ]
            self.upper_bounds_ = [
                medians[j] + self.upper * (q75[j] - q25[j])
                for j in range(m)
            ]

    def transform(self, X: List[List[float]]) -> List[List[float]]:
        """Clip values to the learned bounds.

        Parameters
        ----------
        X : List[List[float]]
            Data to transform (n_samples x n_features).

        Returns
        -------
        List[List[float]]
            Winsorized data with values clipped to bounds.

        Raises
        ------
        RuntimeError
            If transform is called before fit.
        """
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise RuntimeError("Must call fit() before transform().")
        m = len(self.lower_bounds_)
        result: List[List[float]] = []
        for row in X:
            new_row: List[float] = []
            for j in range(m):
                val = row[j]
                if val < self.lower_bounds_[j]:
                    val = self.lower_bounds_[j]
                elif val > self.upper_bounds_[j]:
                    val = self.upper_bounds_[j]
                new_row.append(val)
            result.append(new_row)
        return result

    def inverse_transform(
        self, X: List[List[float]]
    ) -> List[List[float]]:
        """Inverse transform is not exactly reversible for winsorization.

        Returns data unchanged, since clipped values cannot be recovered.

        Parameters
        ----------
        X : List[List[float]]
            Winsorized data.

        Returns
        -------
        List[List[float]]
            Data unchanged (winsorization is not invertible).
        """
        return [row[:] for row in X]
