"""
Weight of Evidence (WoE) and Information Value (IV) Binning

Implements WoE transformation for credit scorecard development.
WoE converts categorical/continuous features into a scale that
measures the relative risk (log-odds) of each bin, enabling
transparent and interpretable credit scoring models.

IV interpretation thresholds (industry standard):
    - < 0.02:  Useless predictor
    - 0.02-0.1: Weak predictor
    - 0.1-0.3: Medium predictor
    - 0.3-0.5: Strong predictor
    - > 0.5:  Suspicious (likely overfitting)

All computations are pure Python with no external dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class BinResult:
    """Result of WoE binning analysis.

    Attributes
    ----------
    boundaries : List[float]
        Bin boundary values (n_bins + 1 edges).
    woe_values : List[float]
        Weight of Evidence for each bin.
    iv_values : List[float]
        Information Value contribution of each bin.
    total_iv : float
        Total Information Value across all bins.
    event_rates : List[float]
        Event rate (proportion of events) in each bin.
    counts : List[int]
        Total observation count in each bin.
    non_event_counts : List[int]
        Non-event count in each bin.
    """

    boundaries: List[float]
    woe_values: List[float]
    iv_values: List[float]
    total_iv: float
    event_rates: List[float]
    counts: List[int]
    non_event_counts: List[int]


class WoEBinner:
    """Weight of Evidence binner for credit scorecard features.

    Bins a continuous feature and computes WoE for each bin.
    Supports quantile and equal-width binning, optional monotonic
    WoE enforcement, and Laplace smoothing for zero-count bins.

    Parameters
    ----------
    n_bins : int
        Number of initial bins. Default is 10.
    method : str
        Binning method: ``"quantile"`` (default) or
        ``"equal_width"``.
    min_bin_size : float
        Minimum bin size as a fraction of total observations.
        Bins below this threshold are merged. Default is 0.05.
    monotonic : bool
        If True, enforce monotonic WoE by merging adjacent bins
        that violate monotonicity. Default is True.

    Examples
    --------
    >>> binner = WoEBinner(n_bins=5, monotonic=True)
    >>> result = binner.fit(feature_values, target_labels)  # doctest: +SKIP
    >>> woe_transformed = binner.transform(feature_values)  # doctest: +SKIP
    """

    def __init__(
        self,
        n_bins: int = 10,
        method: str = "quantile",
        min_bin_size: float = 0.05,
        monotonic: bool = True,
    ) -> None:
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        if method not in ("quantile", "equal_width"):
            raise ValueError(
                f"method must be 'quantile' or 'equal_width', "
                f"got '{method}'"
            )
        if not 0.0 < min_bin_size < 1.0:
            raise ValueError(
                "min_bin_size must be between 0 and 1"
            )

        self.n_bins = n_bins
        self.method = method
        self.min_bin_size = min_bin_size
        self.monotonic = monotonic

        self._bin_result: Optional[BinResult] = None

    def _compute_quantile_boundaries(
        self, feature: List[float]
    ) -> List[float]:
        """Compute bin boundaries using quantiles.

        Parameters
        ----------
        feature : List[float]
            Feature values.

        Returns
        -------
        List[float]
            Sorted unique bin boundaries.
        """
        sorted_vals = sorted(feature)
        n = len(sorted_vals)
        boundaries = [sorted_vals[0]]

        for i in range(1, self.n_bins):
            q = i / self.n_bins
            pos = q * (n - 1)
            lo = int(math.floor(pos))
            hi = min(lo + 1, n - 1)
            frac = pos - lo
            val = sorted_vals[lo] * (1.0 - frac) + (
                sorted_vals[hi] * frac
            )
            if val != boundaries[-1]:
                boundaries.append(val)

        boundaries.append(sorted_vals[-1])

        # Ensure unique boundaries
        unique: List[float] = [boundaries[0]]
        for b in boundaries[1:]:
            if b > unique[-1]:
                unique.append(b)
        return unique

    def _compute_equal_width_boundaries(
        self, feature: List[float]
    ) -> List[float]:
        """Compute bin boundaries using equal width.

        Parameters
        ----------
        feature : List[float]
            Feature values.

        Returns
        -------
        List[float]
            Equally spaced bin boundaries.
        """
        min_val = min(feature)
        max_val = max(feature)
        step = (max_val - min_val) / self.n_bins
        if step == 0.0:
            return [min_val, max_val]
        boundaries = [
            min_val + i * step for i in range(self.n_bins + 1)
        ]
        boundaries[-1] = max_val
        return boundaries

    def _assign_bins(
        self, feature: List[float], boundaries: List[float]
    ) -> List[int]:
        """Assign each observation to a bin index.

        Parameters
        ----------
        feature : List[float]
            Feature values.
        boundaries : List[float]
            Bin boundaries (n_bins + 1 edges).

        Returns
        -------
        List[int]
            Bin index for each observation.
        """
        n_bins = len(boundaries) - 1
        assignments: List[int] = []
        for val in feature:
            # Binary search for bin
            lo, hi = 0, n_bins - 1
            bin_idx = n_bins - 1  # default to last bin
            while lo <= hi:
                mid = (lo + hi) // 2
                if val < boundaries[mid]:
                    hi = mid - 1
                elif mid < n_bins - 1 and val >= boundaries[mid + 1]:
                    lo = mid + 1
                else:
                    bin_idx = mid
                    break
            assignments.append(bin_idx)
        return assignments

    def _compute_woe_iv(
        self,
        bin_assignments: List[int],
        target: List[int],
        n_bins: int,
    ) -> Tuple[
        List[float],
        List[float],
        List[float],
        List[int],
        List[int],
    ]:
        """Compute WoE and IV for assigned bins.

        Uses Laplace smoothing (add 0.5) for bins with zero
        events or non-events to avoid log(0).

        Parameters
        ----------
        bin_assignments : List[int]
            Bin index for each observation.
        target : List[int]
            Binary target values (0 or 1).
        n_bins : int
            Number of bins.

        Returns
        -------
        Tuple
            (woe_values, iv_values, event_rates, counts,
             non_event_counts)
        """
        # Count events and non-events per bin
        events = [0] * n_bins
        non_events = [0] * n_bins
        counts = [0] * n_bins

        for bin_idx, y in zip(bin_assignments, target):
            counts[bin_idx] += 1
            if y == 1:
                events[bin_idx] += 1
            else:
                non_events[bin_idx] += 1

        total_events = sum(events)
        total_non_events = sum(non_events)

        woe_values: List[float] = []
        iv_values: List[float] = []
        event_rates: List[float] = []

        for i in range(n_bins):
            # Laplace smoothing for zero bins
            ev = events[i] + 0.5
            nev = non_events[i] + 0.5

            pct_events = ev / (total_events + 0.5 * n_bins)
            pct_non_events = nev / (
                total_non_events + 0.5 * n_bins
            )

            woe = math.log(pct_non_events / pct_events)
            iv = (pct_non_events - pct_events) * woe

            woe_values.append(woe)
            iv_values.append(iv)

            if counts[i] > 0:
                event_rates.append(events[i] / counts[i])
            else:
                event_rates.append(0.0)

        return (
            woe_values,
            iv_values,
            event_rates,
            counts,
            [non_events[i] for i in range(n_bins)],
        )

    def _enforce_monotonic(
        self,
        boundaries: List[float],
        woe_values: List[float],
        iv_values: List[float],
        event_rates: List[float],
        counts: List[int],
        non_event_counts: List[int],
        bin_assignments: List[int],
        target: List[int],
    ) -> Tuple[
        List[float],
        List[float],
        List[float],
        List[float],
        List[int],
        List[int],
    ]:
        """Merge adjacent bins to enforce monotonic WoE.

        Iteratively merges the pair of adjacent bins whose WoE
        values most violate monotonicity until the WoE sequence
        is monotonic (either increasing or decreasing).

        Parameters
        ----------
        boundaries : List[float]
            Current bin boundaries.
        woe_values : List[float]
            Current WoE values.
        iv_values : List[float]
            Current IV values.
        event_rates : List[float]
            Current event rates.
        counts : List[int]
            Current bin counts.
        non_event_counts : List[int]
            Current non-event counts.
        bin_assignments : List[int]
            Bin assignment for each observation.
        target : List[int]
            Binary target values.

        Returns
        -------
        Tuple
            Updated (boundaries, woe_values, iv_values,
            event_rates, counts, non_event_counts).
        """
        cur_bounds = boundaries[:]
        cur_woe = woe_values[:]
        cur_iv = iv_values[:]
        cur_er = event_rates[:]
        cur_counts = counts[:]
        cur_ne = non_event_counts[:]

        max_merges = len(cur_woe) - 2  # keep at least 2 bins

        for _ in range(max_merges):
            n_bins = len(cur_woe)
            if n_bins <= 2:
                break

            # Check if already monotonic
            increasing = all(
                cur_woe[i] <= cur_woe[i + 1]
                for i in range(n_bins - 1)
            )
            decreasing = all(
                cur_woe[i] >= cur_woe[i + 1]
                for i in range(n_bins - 1)
            )

            if increasing or decreasing:
                break

            # Determine intended direction by correlation
            # between bin index and WoE
            mean_idx = (n_bins - 1) / 2.0
            mean_woe = sum(cur_woe) / n_bins
            cov_val = sum(
                (i - mean_idx) * (w - mean_woe)
                for i, w in enumerate(cur_woe)
            )

            # Find worst violating pair and merge
            worst_idx = -1
            worst_diff = 0.0

            for i in range(n_bins - 1):
                diff = cur_woe[i + 1] - cur_woe[i]
                if cov_val >= 0:
                    # Should be increasing
                    if diff < worst_diff:
                        worst_diff = diff
                        worst_idx = i
                else:
                    # Should be decreasing
                    if diff > worst_diff:
                        worst_diff = diff
                        worst_idx = i

            if worst_idx == -1:
                break

            # Merge bins worst_idx and worst_idx + 1
            j = worst_idx
            new_bounds = (
                cur_bounds[: j + 1]
                + cur_bounds[j + 2:]
            )

            # Recompute for merged bin
            merged_non_events = cur_ne[j] + cur_ne[j + 1]
            merged_count = cur_counts[j] + cur_counts[j + 1]

            # The original features are not available here, so merge
            # the bin statistics directly
            cur_bounds = new_bounds
            n_b = len(cur_bounds) - 1

            new_counts = (
                cur_counts[:j]
                + [merged_count]
                + cur_counts[j + 2:]
            )
            new_ne = (
                cur_ne[:j]
                + [merged_non_events]
                + cur_ne[j + 2:]
            )

            total_ev = sum(
                new_counts[i] - new_ne[i]
                for i in range(n_b)
            )
            total_nev = sum(new_ne)

            new_woe: List[float] = []
            new_iv: List[float] = []
            new_er: List[float] = []

            for i in range(n_b):
                ev = (new_counts[i] - new_ne[i]) + 0.5
                nev = new_ne[i] + 0.5
                pct_ev = ev / (total_ev + 0.5 * n_b)
                pct_nev = nev / (total_nev + 0.5 * n_b)
                woe = math.log(pct_nev / pct_ev)
                iv = (pct_nev - pct_ev) * woe
                new_woe.append(woe)
                new_iv.append(iv)
                if new_counts[i] > 0:
                    er = (new_counts[i] - new_ne[i]) / (
                        new_counts[i]
                    )
                else:
                    er = 0.0
                new_er.append(er)

            cur_woe = new_woe
            cur_iv = new_iv
            cur_er = new_er
            cur_counts = new_counts
            cur_ne = new_ne

        return (
            cur_bounds,
            cur_woe,
            cur_iv,
            cur_er,
            cur_counts,
            cur_ne,
        )

    def fit(
        self, feature: List[float], target: List[int]
    ) -> BinResult:
        """Fit the WoE binner to a feature and target.

        Computes bin boundaries, assigns observations to bins,
        calculates WoE and IV for each bin, and optionally
        enforces monotonic WoE.

        Parameters
        ----------
        feature : List[float]
            Feature values (n_samples,).
        target : List[int]
            Binary target values (0 or 1).

        Returns
        -------
        BinResult
            Binning analysis with WoE, IV, and event rates.

        Raises
        ------
        ValueError
            If inputs are invalid.
        """
        if len(feature) != len(target):
            raise ValueError(
                "feature and target must have same length"
            )
        if len(feature) == 0:
            raise ValueError("Inputs must not be empty")

        n = len(feature)
        min_count = max(1, int(n * self.min_bin_size))

        # Compute boundaries
        if self.method == "quantile":
            boundaries = self._compute_quantile_boundaries(
                feature
            )
        else:
            boundaries = self._compute_equal_width_boundaries(
                feature
            )

        n_bins = len(boundaries) - 1

        # Assign observations to bins
        bin_assignments = self._assign_bins(feature, boundaries)

        # Compute WoE and IV
        (
            woe_values,
            iv_values,
            event_rates,
            counts,
            non_event_counts,
        ) = self._compute_woe_iv(bin_assignments, target, n_bins)

        # Merge small bins
        merged = True
        while merged:
            merged = False
            for i in range(len(counts)):
                if counts[i] < min_count and len(counts) > 2:
                    # Merge with neighbor that has fewer obs
                    if i == 0:
                        merge_with = 1
                    elif i == len(counts) - 1:
                        merge_with = i - 1
                    else:
                        if counts[i - 1] <= counts[i + 1]:
                            merge_with = i - 1
                        else:
                            merge_with = i + 1

                    lo = min(i, merge_with)
                    hi = max(i, merge_with)

                    # Merge bins lo and hi
                    boundaries = (
                        boundaries[: lo + 1]
                        + boundaries[hi + 1:]
                    )

                    n_bins = len(boundaries) - 1
                    bin_assignments = self._assign_bins(
                        feature, boundaries
                    )
                    (
                        woe_values,
                        iv_values,
                        event_rates,
                        counts,
                        non_event_counts,
                    ) = self._compute_woe_iv(
                        bin_assignments, target, n_bins
                    )
                    merged = True
                    break

        # Enforce monotonic WoE
        if self.monotonic and len(woe_values) > 2:
            (
                boundaries,
                woe_values,
                iv_values,
                event_rates,
                counts,
                non_event_counts,
            ) = self._enforce_monotonic(
                boundaries,
                woe_values,
                iv_values,
                event_rates,
                counts,
                non_event_counts,
                bin_assignments,
                target,
            )

        total_iv = sum(iv_values)

        self._bin_result = BinResult(
            boundaries=boundaries,
            woe_values=woe_values,
            iv_values=iv_values,
            total_iv=total_iv,
            event_rates=event_rates,
            counts=counts,
            non_event_counts=non_event_counts,
        )

        return self._bin_result

    def transform(self, feature: List[float]) -> List[float]:
        """Transform feature values to WoE.

        Replaces each feature value with the WoE of its
        assigned bin.

        Parameters
        ----------
        feature : List[float]
            Feature values (n_samples,).

        Returns
        -------
        List[float]
            WoE-transformed values.

        Raises
        ------
        RuntimeError
            If the binner has not been fitted.
        """
        if self._bin_result is None:
            raise RuntimeError("WoEBinner has not been fitted")

        result = self._bin_result
        bin_assignments = self._assign_bins(
            feature, result.boundaries
        )

        return [result.woe_values[b] for b in bin_assignments]

    def fit_transform(
        self, feature: List[float], target: List[int]
    ) -> List[float]:
        """Fit the binner and transform in one step.

        Parameters
        ----------
        feature : List[float]
            Feature values (n_samples,).
        target : List[int]
            Binary target values (0 or 1).

        Returns
        -------
        List[float]
            WoE-transformed feature values.
        """
        self.fit(feature, target)
        return self.transform(feature)
