"""
Isolation Forest Anomaly Detection

Implementation of the Isolation Forest algorithm following
Liu, Ting, and Zhou (2008). Anomaly scores are computed from
average path lengths in an ensemble of isolation trees.
All operations use pure Python with no NumPy dependency.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional

from finceptquantlib.ml._base import Clusterer


# Euler-Mascheroni constant
_EULER_MASCHERONI: float = 0.5772156649015329


@dataclass
class IsolationForestResult:
    """Result of Isolation Forest anomaly detection.

    Attributes
    ----------
    anomaly_scores : List[float]
        Anomaly score for each sample in [0, 1].
        Higher values indicate more anomalous points.
    labels : List[int]
        ``-1`` for anomaly, ``1`` for normal.
    threshold : float
        Decision threshold derived from the contamination
        parameter.
    n_anomalies : int
        Number of detected anomalies.
    """

    anomaly_scores: List[float]
    labels: List[int]
    threshold: float
    n_anomalies: int


class _IsolationTreeNode:
    """Internal node of an isolation tree.

    Attributes
    ----------
    split_feature : int or None
        Feature index used for splitting (``None`` for leaves).
    split_value : float
        Threshold value for the split.
    left : _IsolationTreeNode or None
        Left child (values < split_value).
    right : _IsolationTreeNode or None
        Right child (values >= split_value).
    size : int
        Number of samples that reached this node.
    """

    __slots__ = (
        "split_feature",
        "split_value",
        "left",
        "right",
        "size",
    )

    def __init__(self) -> None:
        self.split_feature: Optional[int] = None
        self.split_value: float = 0.0
        self.left: Optional[_IsolationTreeNode] = None
        self.right: Optional[_IsolationTreeNode] = None
        self.size: int = 0


class IsolationForest(Clusterer):
    """Isolation Forest for anomaly detection.

    Based on Liu, Ting, and Zhou (2008). Anomalies are
    isolated in fewer splits on average, yielding shorter
    path lengths in random trees.

    Parameters
    ----------
    n_estimators : int
        Number of isolation trees in the ensemble.
    max_samples : int
        Number of samples drawn (without replacement) to
        build each tree.
    contamination : float
        Expected fraction of anomalies in [0, 0.5]. Used
        to set the decision threshold.
    random_state : int
        Seed for reproducibility.

    Examples
    --------
    >>> from finceptquantlib.ml.modeling.clustering import IsolationForest
    >>> X = [[1.0, 2.0]] * 50 + [[100.0, 200.0]]
    >>> iso = IsolationForest(
    ...     n_estimators=50, contamination=0.05, random_state=0
    ... )
    >>> result = iso.fit(X)
    >>> result.labels[-1]
    -1
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int = 256,
        contamination: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self._trees: List[_IsolationTreeNode] = []
        self._effective_max_samples: int = max_samples

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self, X: List[List[float]]
    ) -> IsolationForestResult:
        """Build the isolation forest and score training data.

        Parameters
        ----------
        X : List[List[float]]
            Data matrix (n_samples x n_features).

        Returns
        -------
        IsolationForestResult
            Anomaly scores, labels, threshold, and anomaly count.
        """
        n = len(X)
        rng = random.Random(self.random_state)
        self._effective_max_samples = min(self.max_samples, n)
        max_depth = int(math.ceil(
            math.log2(max(self._effective_max_samples, 2))
        ))

        self._trees = []
        for _ in range(self.n_estimators):
            indices = rng.sample(
                range(n), self._effective_max_samples
            )
            subsample = [X[i] for i in indices]
            tree = self._build_tree(
                subsample, 0, max_depth, rng
            )
            self._trees.append(tree)

        scores = self.anomaly_score(X)
        threshold = self._compute_threshold(scores)
        labels = [
            -1 if s >= threshold else 1 for s in scores
        ]
        n_anomalies = labels.count(-1)

        return IsolationForestResult(
            anomaly_scores=scores,
            labels=labels,
            threshold=threshold,
            n_anomalies=n_anomalies,
        )

    def predict(self, X: List[List[float]]) -> List[int]:
        """Predict anomaly labels for new data.

        Parameters
        ----------
        X : List[List[float]]
            Data matrix (n_samples x n_features).

        Returns
        -------
        List[int]
            ``-1`` for anomaly, ``1`` for normal.

        Raises
        ------
        RuntimeError
            If the model has not been fit.
        """
        if not self._trees:
            raise RuntimeError("Model has not been fit yet.")
        scores = self.anomaly_score(X)
        threshold = self._compute_threshold(scores)
        return [-1 if s >= threshold else 1 for s in scores]

    def anomaly_score(
        self, X: List[List[float]]
    ) -> List[float]:
        """Compute anomaly scores for data points.

        Score is ``s(x, n) = 2^{-E[h(x)] / c(n)}``, following
        Liu, Ting, Zhou (2008), where ``h(x)`` is the path
        length and ``c(n)`` is the average path length of an
        unsuccessful search in a BST.

        Parameters
        ----------
        X : List[List[float]]
            Data matrix (n_samples x n_features).

        Returns
        -------
        List[float]
            Anomaly scores in [0, 1]. Higher means more
            anomalous.

        Raises
        ------
        RuntimeError
            If the model has not been fit.
        """
        if not self._trees:
            raise RuntimeError("Model has not been fit yet.")

        psi = self._effective_max_samples
        c_n = self._average_path_length(psi)

        scores: List[float] = []
        for x in X:
            mean_path = sum(
                self._path_length(x, tree, 0)
                for tree in self._trees
            ) / len(self._trees)

            if c_n == 0.0:
                score = 0.5
            else:
                score = 2.0 ** (-mean_path / c_n)
            scores.append(score)
        return scores

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_tree(
        self,
        X: List[List[float]],
        depth: int,
        max_depth: int,
        rng: random.Random,
    ) -> _IsolationTreeNode:
        """Recursively build an isolation tree.

        Parameters
        ----------
        X : List[List[float]]
            Current subsample.
        depth : int
            Current depth.
        max_depth : int
            Maximum tree depth (ceil(log2(max_samples))).
        rng : random.Random
            Random number generator.

        Returns
        -------
        _IsolationTreeNode
            Root of the (sub)tree.
        """
        node = _IsolationTreeNode()
        node.size = len(X)

        if len(X) <= 1 or depth >= max_depth:
            return node

        m = len(X[0])
        # Random feature
        feat = rng.randrange(m)

        # Min / max of feature in current data
        feat_min = X[0][feat]
        feat_max = X[0][feat]
        for row in X[1:]:
            v = row[feat]
            if v < feat_min:
                feat_min = v
            if v > feat_max:
                feat_max = v

        if feat_min == feat_max:
            # All values identical on this feature; leaf
            return node

        split_val = rng.uniform(feat_min, feat_max)
        node.split_feature = feat
        node.split_value = split_val

        left_data: List[List[float]] = []
        right_data: List[List[float]] = []
        for row in X:
            if row[feat] < split_val:
                left_data.append(row)
            else:
                right_data.append(row)

        node.left = self._build_tree(
            left_data, depth + 1, max_depth, rng
        )
        node.right = self._build_tree(
            right_data, depth + 1, max_depth, rng
        )
        return node

    def _path_length(
        self,
        x: List[float],
        node: _IsolationTreeNode,
        depth: int,
    ) -> float:
        """Compute path length for a single point in a tree.

        At external nodes, the path length is adjusted by
        ``c(node.size)`` to account for unbuilt subtrees.

        Parameters
        ----------
        x : List[float]
            Data point.
        node : _IsolationTreeNode
            Current tree node.
        depth : int
            Current depth.

        Returns
        -------
        float
            Adjusted path length.
        """
        if node.split_feature is None:
            # External node: adjust for unbuilt subtree
            return float(depth) + self._average_path_length(
                node.size
            )

        if x[node.split_feature] < node.split_value:
            return self._path_length(
                x, node.left, depth + 1  # type: ignore[arg-type]
            )
        else:
            return self._path_length(
                x, node.right, depth + 1  # type: ignore[arg-type]
            )

    @staticmethod
    def _average_path_length(n: int) -> float:
        """Average path length of unsuccessful search in BST.

        Follows Liu, Ting, Zhou (2008):
            c(n) = 2 * H(n-1) - 2*(n-1)/n

        where ``H(k) = ln(k) + Euler-Mascheroni``.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        float
            Expected path length ``c(n)``.
        """
        if n <= 1:
            return 0.0
        if n == 2:
            return 1.0
        h = math.log(n - 1) + _EULER_MASCHERONI
        return 2.0 * h - 2.0 * (n - 1) / n

    def _compute_threshold(
        self, scores: List[float]
    ) -> float:
        """Determine anomaly threshold from contamination.

        Parameters
        ----------
        scores : List[float]
            Anomaly scores for all samples.

        Returns
        -------
        float
            Score threshold: points scoring at or above this
            value are classified as anomalies.
        """
        sorted_scores = sorted(scores)
        n = len(sorted_scores)
        # We want the top `contamination` fraction flagged
        idx = max(
            0,
            int(math.ceil((1.0 - self.contamination) * n)) - 1,
        )
        idx = min(idx, n - 1)
        return sorted_scores[idx]
