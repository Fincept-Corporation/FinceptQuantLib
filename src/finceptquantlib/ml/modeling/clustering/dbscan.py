"""
DBSCAN Density-Based Clustering

Density-Based Spatial Clustering of Applications with Noise.
Identifies clusters as dense regions separated by low-density
areas, with noise points labelled ``-1``.
All operations use pure Python with no NumPy dependency.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import List

from finceptquantlib.ml._base import Clusterer
from finceptquantlib.ml._matrix_utils import (
    distance_matrix,
    euclidean_distance,
)


@dataclass
class DBSCANResult:
    """Result of DBSCAN clustering.

    Attributes
    ----------
    labels : List[int]
        Cluster label for each sample. ``-1`` indicates noise.
    n_clusters : int
        Number of clusters found (excluding noise).
    core_sample_indices : List[int]
        Indices of core points in the dataset.
    n_noise : int
        Number of noise points.
    """

    labels: List[int]
    n_clusters: int
    core_sample_indices: List[int]
    n_noise: int


class DBSCAN(Clusterer):
    """DBSCAN density-based clustering.

    A point is a *core point* if at least ``min_samples``
    neighbors lie within distance ``eps``. Clusters are formed
    by connecting overlapping core-point neighborhoods. Points
    reachable from a core point but not themselves core are
    *border* points. All other points are *noise*.

    Parameters
    ----------
    eps : float
        Maximum distance for two points to be considered
        neighbors.
    min_samples : int
        Minimum number of neighbors (including the point
        itself) required to form a core point.
    metric : str
        Distance metric. Currently only ``"euclidean"`` is
        supported.

    Examples
    --------
    >>> from finceptquantlib.ml.modeling.clustering import DBSCAN
    >>> X = [[0.0, 0.0], [0.1, 0.1], [0.2, 0.0],
    ...      [10.0, 10.0], [10.1, 10.1], [10.2, 10.0]]
    >>> db = DBSCAN(eps=0.5, min_samples=2)
    >>> result = db.fit(X)
    >>> result.n_clusters
    2
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
    ) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self._core_points: List[List[float]] = []
        self._core_labels: List[int] = []
        self._labels: List[int] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: List[List[float]]) -> DBSCANResult:
        """Fit DBSCAN to data.

        Parameters
        ----------
        X : List[List[float]]
            Data matrix (n_samples x n_features).

        Returns
        -------
        DBSCANResult
            Clustering result with labels, cluster count,
            core indices, and noise count.
        """
        n = len(X)
        dist = distance_matrix(X)

        # Find neighbors for each point
        neighbors: List[List[int]] = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if dist[i][j] <= self.eps:
                    neighbors[i].append(j)

        # Identify core points
        is_core = [
            len(neighbors[i]) >= self.min_samples
            for i in range(n)
        ]
        core_indices = [i for i in range(n) if is_core[i]]

        # BFS cluster expansion
        labels = [-1] * n
        cluster_id = 0

        for i in range(n):
            if not is_core[i]:
                continue
            if labels[i] != -1:
                continue

            # Start a new cluster from core point i
            queue: deque[int] = deque([i])
            labels[i] = cluster_id

            while queue:
                current = queue.popleft()
                for nb in neighbors[current]:
                    if labels[nb] == -1:
                        labels[nb] = cluster_id
                        # Only expand from core points
                        if is_core[nb]:
                            queue.append(nb)

            cluster_id += 1

        # Store state for predict
        self._core_points = [X[i] for i in core_indices]
        self._core_labels = [labels[i] for i in core_indices]
        self._labels = labels

        n_noise = labels.count(-1)
        return DBSCANResult(
            labels=labels,
            n_clusters=cluster_id,
            core_sample_indices=core_indices,
            n_noise=n_noise,
        )

    def predict(self, X: List[List[float]]) -> List[int]:
        """Assign new points to nearest core point's cluster.

        Points farther than ``eps`` from all core points are
        labelled ``-1`` (noise).

        Parameters
        ----------
        X : List[List[float]]
            Data matrix (n_samples x n_features).

        Returns
        -------
        List[int]
            Cluster labels (``-1`` for noise).

        Raises
        ------
        RuntimeError
            If the model has not been fit.
        """
        if not self._core_points and self._labels:
            # Model was fit but had no core points
            return [-1] * len(X)
        if not self._core_points:
            raise RuntimeError("Model has not been fit yet.")

        labels: List[int] = []
        for x in X:
            best_label = -1
            best_dist = math.inf
            for idx, core_pt in enumerate(self._core_points):
                d = euclidean_distance(x, core_pt)
                if d <= self.eps and d < best_dist:
                    best_dist = d
                    best_label = self._core_labels[idx]
            labels.append(best_label)
        return labels
