"""
Missing Value Handler

Provides methods for imputing missing values in financial time-series
and cross-sectional data. Missing values are represented as ``None``
in ``Optional[float]`` lists, or as ``float('nan')``.

Supported imputation methods:

- FORWARD_FILL: Propagate last valid observation forward
- BACKWARD_FILL: Propagate next valid observation backward
- LINEAR_INTERPOLATION: Linearly interpolate between valid neighbors
- MEDIAN: Replace with column median from training data
- MEAN: Replace with column mean from training data

For time-series data, forward fill and linear interpolation are
preferred. An optional ``max_gap`` parameter limits how many
consecutive missing values can be filled.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import List, Optional

from finceptquantlib.ml._matrix_utils import column_mean, column_median


class MissingValueMethod(Enum):
    """Imputation strategy for missing values.

    Attributes
    ----------
    FORWARD_FILL
        Carry the last valid observation forward.
    BACKWARD_FILL
        Carry the next valid observation backward.
    LINEAR_INTERPOLATION
        Linearly interpolate between surrounding valid values.
    MEDIAN
        Replace with column median learned during fit.
    MEAN
        Replace with column mean learned during fit.
    """

    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    LINEAR_INTERPOLATION = "linear_interpolation"
    MEDIAN = "median"
    MEAN = "mean"


def _is_missing(val: Optional[float]) -> bool:
    """Check whether a value is missing (None or NaN).

    Parameters
    ----------
    val : Optional[float]
        Value to check.

    Returns
    -------
    bool
        True if val is None or NaN.
    """
    if val is None:
        return True
    try:
        return math.isnan(val)
    except (TypeError, ValueError):
        return False


class MissingValueHandler:
    """Impute missing values in tabular data.

    Works with ``List[List[Optional[float]]]`` where ``None`` or
    ``float('nan')`` represents a missing value. For MEDIAN and MEAN
    methods, ``fit()`` must be called first to learn fill values from
    complete observations.

    Parameters
    ----------
    method : str
        Imputation method. One of ``"forward_fill"``,
        ``"backward_fill"``, ``"linear_interpolation"``,
        ``"median"``, or ``"mean"``.
    max_gap : Optional[int]
        Maximum number of consecutive missing values to fill.
        If a gap exceeds this, remaining values stay as ``None``.
        Default is None (no limit).

    Attributes
    ----------
    fill_values_ : Optional[List[float]]
        Per-feature fill values learned during fit (MEDIAN/MEAN only).

    Examples
    --------
    >>> handler = MissingValueHandler(method="forward_fill")
    >>> data = [[1.0, None], [None, 2.0], [3.0, None]]
    >>> handler.fit(data)
    >>> filled = handler.transform(data)
    """

    def __init__(
        self,
        method: str = "forward_fill",
        max_gap: Optional[int] = None,
    ) -> None:
        try:
            self.method = MissingValueMethod(method)
        except ValueError:
            valid = [m.value for m in MissingValueMethod]
            raise ValueError(
                f"Unknown method '{method}'. Expected one of {valid}."
            ) from None
        self.max_gap = max_gap
        self.fill_values_: Optional[List[float]] = None

    def fit(
        self, X: List[List[Optional[float]]]
    ) -> None:
        """Learn fill values from training data.

        For MEDIAN and MEAN methods, computes per-feature statistics
        from non-missing observations. For other methods, this is a
        no-op but must still be called for API consistency.

        Parameters
        ----------
        X : List[List[Optional[float]]]
            Training data with possible missing values.

        Raises
        ------
        ValueError
            If X is empty.
        """
        if not X:
            raise ValueError("Cannot fit on empty data.")

        if self.method in (
            MissingValueMethod.MEDIAN,
            MissingValueMethod.MEAN,
        ):
            m = len(X[0])
            # Extract non-missing values per column
            columns: List[List[float]] = [[] for _ in range(m)]
            for row in X:
                for j in range(m):
                    if not _is_missing(row[j]):
                        columns[j].append(float(row[j]))  # type: ignore[arg-type]

            if self.method == MissingValueMethod.MEAN:
                self.fill_values_ = []
                for j in range(m):
                    if columns[j]:
                        col_as_matrix = [[v] for v in columns[j]]
                        self.fill_values_.append(
                            column_mean(col_as_matrix)[0]
                        )
                    else:
                        self.fill_values_.append(0.0)
            else:  # MEDIAN
                self.fill_values_ = []
                for j in range(m):
                    if columns[j]:
                        col_as_matrix = [[v] for v in columns[j]]
                        self.fill_values_.append(
                            column_median(col_as_matrix)[0]
                        )
                    else:
                        self.fill_values_.append(0.0)

    def transform(
        self, X: List[List[Optional[float]]]
    ) -> List[List[Optional[float]]]:
        """Fill missing values using the configured method.

        Parameters
        ----------
        X : List[List[Optional[float]]]
            Data with possible missing values.

        Returns
        -------
        List[List[Optional[float]]]
            Data with missing values imputed. Values that could not
            be filled (e.g., due to max_gap) remain as None.

        Raises
        ------
        RuntimeError
            If MEDIAN/MEAN method is used without prior fit.
        """
        if not X:
            return []

        m = len(X[0])
        n = len(X)

        if self.method == MissingValueMethod.FORWARD_FILL:
            return self._forward_fill(X, n, m)
        elif self.method == MissingValueMethod.BACKWARD_FILL:
            return self._backward_fill(X, n, m)
        elif self.method == MissingValueMethod.LINEAR_INTERPOLATION:
            return self._linear_interpolation(X, n, m)
        elif self.method in (
            MissingValueMethod.MEDIAN,
            MissingValueMethod.MEAN,
        ):
            return self._fill_constant(X, n, m)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def _forward_fill(
        self,
        X: List[List[Optional[float]]],
        n: int,
        m: int,
    ) -> List[List[Optional[float]]]:
        """Forward-fill missing values column by column.

        Parameters
        ----------
        X : List[List[Optional[float]]]
            Input data.
        n : int
            Number of rows.
        m : int
            Number of columns.

        Returns
        -------
        List[List[Optional[float]]]
            Forward-filled data.
        """
        result: List[List[Optional[float]]] = [
            list(row) for row in X
        ]
        for j in range(m):
            last_valid: Optional[float] = None
            gap_count = 0
            for i in range(n):
                if _is_missing(result[i][j]):
                    gap_count += 1
                    if last_valid is not None:
                        if (
                            self.max_gap is None
                            or gap_count <= self.max_gap
                        ):
                            result[i][j] = last_valid
                else:
                    last_valid = result[i][j]
                    gap_count = 0
        return result

    def _backward_fill(
        self,
        X: List[List[Optional[float]]],
        n: int,
        m: int,
    ) -> List[List[Optional[float]]]:
        """Backward-fill missing values column by column.

        Parameters
        ----------
        X : List[List[Optional[float]]]
            Input data.
        n : int
            Number of rows.
        m : int
            Number of columns.

        Returns
        -------
        List[List[Optional[float]]]
            Backward-filled data.
        """
        result: List[List[Optional[float]]] = [
            list(row) for row in X
        ]
        for j in range(m):
            next_valid: Optional[float] = None
            gap_count = 0
            for i in range(n - 1, -1, -1):
                if _is_missing(result[i][j]):
                    gap_count += 1
                    if next_valid is not None:
                        if (
                            self.max_gap is None
                            or gap_count <= self.max_gap
                        ):
                            result[i][j] = next_valid
                else:
                    next_valid = result[i][j]
                    gap_count = 0
        return result

    def _linear_interpolation(
        self,
        X: List[List[Optional[float]]],
        n: int,
        m: int,
    ) -> List[List[Optional[float]]]:
        """Linearly interpolate missing values column by column.

        For each gap of missing values, interpolates between the
        last valid value before the gap and the first valid value
        after the gap. Edge gaps (at beginning or end) are left
        unfilled.

        Parameters
        ----------
        X : List[List[Optional[float]]]
            Input data.
        n : int
            Number of rows.
        m : int
            Number of columns.

        Returns
        -------
        List[List[Optional[float]]]
            Interpolated data.
        """
        result: List[List[Optional[float]]] = [
            list(row) for row in X
        ]
        for j in range(m):
            i = 0
            while i < n:
                if _is_missing(result[i][j]):
                    # Find start of gap
                    gap_start = i
                    while i < n and _is_missing(result[i][j]):
                        i += 1
                    gap_end = i  # first non-missing after gap

                    gap_len = gap_end - gap_start
                    if (
                        self.max_gap is not None
                        and gap_len > self.max_gap
                    ):
                        continue

                    # Need valid values on both sides
                    if gap_start > 0 and gap_end < n:
                        left_val = float(
                            result[gap_start - 1][j]  # type: ignore[arg-type]
                        )
                        right_val = float(
                            result[gap_end][j]  # type: ignore[arg-type]
                        )
                        total_steps = gap_len + 1
                        for k in range(gap_len):
                            frac = (k + 1) / total_steps
                            result[gap_start + k][j] = (
                                left_val + frac * (right_val - left_val)
                            )
                else:
                    i += 1
        return result

    def _fill_constant(
        self,
        X: List[List[Optional[float]]],
        n: int,
        m: int,
    ) -> List[List[Optional[float]]]:
        """Fill missing values with pre-computed constants.

        Parameters
        ----------
        X : List[List[Optional[float]]]
            Input data.
        n : int
            Number of rows.
        m : int
            Number of columns.

        Returns
        -------
        List[List[Optional[float]]]
            Data with missing values replaced by fill constants.

        Raises
        ------
        RuntimeError
            If fill_values_ have not been learned via fit().
        """
        if self.fill_values_ is None:
            raise RuntimeError(
                "Must call fit() before transform() for "
                f"{self.method.value} method."
            )
        result: List[List[Optional[float]]] = [
            list(row) for row in X
        ]
        for j in range(m):
            gap_count = 0
            for i in range(n):
                if _is_missing(result[i][j]):
                    gap_count += 1
                    if (
                        self.max_gap is None
                        or gap_count <= self.max_gap
                    ):
                        result[i][j] = self.fill_values_[j]
                else:
                    gap_count = 0
        return result
