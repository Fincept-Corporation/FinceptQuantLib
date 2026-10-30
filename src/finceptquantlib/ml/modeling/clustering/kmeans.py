"""
K-Means Clustering Algorithm

Implements K-Means clustering with K-Means++ initialization,
multiple restarts, and silhouette score computation.
All operations use pure Python with no NumPy dependency.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from finceptquantlib.ml._base import Clusterer
from finceptquantlib.ml._matrix_utils import euclidean_distance


@dataclass
class KMeansResult:
    """Result of K-Means clustering.

    Attributes
    ----------
    labels : List[int]
        Cluster assignment for each sample.
    centroids : List[List[float]]
        Final centroid coordinates (n_clusters x n_features).
    inertia : float
        Sum of squared distances to nearest centroid.
    n_iterations : int
        Number of iterations until convergence.
    silhouette_score : float
        Mean silhouette coefficient over all samples.
    """

    labels: List[int]
    centroids: List[List[float]]
    inertia: float
    n_iterations: int
    silhouette_score: float


class KMeans(Clusterer):
    """K-Means clustering with K-Means++ initialization.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to form.
    max_iter : int
        Maximum number of Lloyd's algorithm iterations.
    tol : float
        Convergence tolerance on centroid movement.
    n_init : int
        Number of independent runs with different initializations.
        The run with lowest inertia is kept.
    init : str
        Initialization method. ``"kmeans++"`` for K-Means++,
        ``"random"`` for random selection from data.
    random_state : int
        Seed for reproducibility.

    Examples
    --------
    >>> from finceptquantlib.ml.modeling.clustering import KMeans
    >>> X = [[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0]]
    >>> km = KMeans(n_clusters=2, random_state=42)
    >>> result = km.fit(X)
    >>> result.labels
    [0, 0, 1, 1]
    """

    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 300,
        tol: float = 1e-4,
        n_init: int = 10,
        init: str = "kmeans++",
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.init = init
        self.random_state = random_state
        self._centroids: List[List[float]] = []
        self._labels: List[int] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: List[List[float]]) -> KMeansResult:
        """Fit K-Means to data.

        Parameters
        ----------
        X : List[List[float]]
            Data matrix (n_samples x n_features).

        Returns
        -------
        KMeansResult
            Clustering result with labels, centroids, inertia,
            iteration count, and silhouette score.
        """
        rng = random.Random(self.random_state)
        best_inertia = math.inf
        best_labels: List[int] = []
        best_centroids: List[List[float]] = []
        best_iters = 0

        for _ in range(self.n_init):
            centroids = self._init_centroids(X, rng)
            labels: List[int] = []
            n_iter = 0

            for n_iter in range(1, self.max_iter + 1):
                labels = self._assign(X, centroids)
                new_centroids = self._update_centroids(
                    X, labels, centroids
                )
                shift = self._centroid_shift(centroids, new_centroids)
                centroids = new_centroids
                if shift < self.tol:
                    break

            inertia = self._compute_inertia(X, labels, centroids)
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centroids = centroids
                best_iters = n_iter

        self._centroids = best_centroids
        self._labels = best_labels

        sil = self._silhouette(X, best_labels)
        return KMeansResult(
            labels=best_labels,
            centroids=best_centroids,
            inertia=best_inertia,
            n_iterations=best_iters,
            silhouette_score=sil,
        )

    def predict(self, X: List[List[float]]) -> List[int]:
        """Assign new points to nearest centroid.

        Parameters
        ----------
        X : List[List[float]]
            Data matrix (n_samples x n_features).

        Returns
        -------
        List[int]
            Cluster label for each sample.

        Raises
        ------
        RuntimeError
            If the model has not been fit.
        """
        if not self._centroids:
            raise RuntimeError("Model has not been fit yet.")
        return self._assign(X, self._centroids)

    def elbow_scores(
        self,
        X: List[List[float]],
        k_range: List[int],
    ) -> List[Tuple[int, float]]:
        """Compute inertia for a range of cluster counts.

        Useful for the elbow method to select the optimal k.

        Parameters
        ----------
        X : List[List[float]]
            Data matrix (n_samples x n_features).
        k_range : List[int]
            List of k values to evaluate.

        Returns
        -------
        List[Tuple[int, float]]
            Pairs of (k, inertia) for each requested k.
        """
        results: List[Tuple[int, float]] = []
        for k in k_range:
            km = KMeans(
                n_clusters=k,
                max_iter=self.max_iter,
                tol=self.tol,
                n_init=self.n_init,
                init=self.init,
                random_state=self.random_state,
            )
            res = km.fit(X)
            results.append((k, res.inertia))
        return results

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model parameters to dictionary.

        Returns
        -------
        Dict[str, Any]
            JSON-serializable model state.
        """
        return {
            "n_clusters": self.n_clusters,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "n_init": self.n_init,
            "init": self.init,
            "random_state": self.random_state,
            "centroids": self._centroids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KMeans":
        """Reconstruct model from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Serialized model state.

        Returns
        -------
        KMeans
            Restored model instance.
        """
        km = cls(
            n_clusters=data["n_clusters"],
            max_iter=data["max_iter"],
            tol=data["tol"],
            n_init=data["n_init"],
            init=data["init"],
            random_state=data["random_state"],
        )
        km._centroids = data.get("centroids", [])
        return km

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_centroids(
        self,
        X: List[List[float]],
        rng: random.Random,
    ) -> List[List[float]]:
        """Initialize centroids using K-Means++ or random selection.

        Parameters
        ----------
        X : List[List[float]]
            Data matrix.
        rng : random.Random
            Random number generator instance.

        Returns
        -------
        List[List[float]]
            Initial centroid positions.
        """
        n = len(X)
        if self.init == "random":
            indices = rng.sample(range(n), min(self.n_clusters, n))
            return [list(X[i]) for i in indices]

        # K-Means++ initialization
        centroids: List[List[float]] = [list(X[rng.randrange(n)])]

        for _ in range(1, self.n_clusters):
            dists = [
                min(
                    self._sq_dist(x, c) for c in centroids
                )
                for x in X
            ]
            total = sum(dists)
            if total == 0.0:
                # All remaining points coincide with a centroid
                idx = rng.randrange(n)
            else:
                r = rng.random() * total
                cumulative = 0.0
                idx = 0
                for i, d in enumerate(dists):
                    cumulative += d
                    if cumulative >= r:
                        idx = i
                        break
            centroids.append(list(X[idx]))

        return centroids

    @staticmethod
    def _sq_dist(a: List[float], b: List[float]) -> float:
        """Squared Euclidean distance."""
        return sum((ai - bi) ** 2 for ai, bi in zip(a, b))

    def _assign(
        self,
        X: List[List[float]],
        centroids: List[List[float]],
    ) -> List[int]:
        """Assign each point to its nearest centroid.

        Parameters
        ----------
        X : List[List[float]]
            Data matrix.
        centroids : List[List[float]]
            Current centroid positions.

        Returns
        -------
        List[int]
            Cluster label per sample.
        """
        labels: List[int] = []
        for x in X:
            best_k = 0
            best_d = math.inf
            for k, c in enumerate(centroids):
                d = self._sq_dist(x, c)
                if d < best_d:
                    best_d = d
                    best_k = k
            labels.append(best_k)
        return labels

    def _update_centroids(
        self,
        X: List[List[float]],
        labels: List[int],
        old_centroids: List[List[float]],
    ) -> List[List[float]]:
        """Recompute centroids as cluster means.

        Parameters
        ----------
        X : List[List[float]]
            Data matrix.
        labels : List[int]
            Current cluster assignments.
        old_centroids : List[List[float]]
            Previous centroids (used for empty clusters).

        Returns
        -------
        List[List[float]]
            Updated centroid positions.
        """
        m = len(X[0])
        counts = [0] * self.n_clusters
        sums = [[0.0] * m for _ in range(self.n_clusters)]
        for i, x in enumerate(X):
            k = labels[i]
            counts[k] += 1
            for j in range(m):
                sums[k][j] += x[j]

        centroids: List[List[float]] = []
        for k in range(self.n_clusters):
            if counts[k] > 0:
                centroids.append(
                    [s / counts[k] for s in sums[k]]
                )
            else:
                # Keep old centroid for empty clusters
                centroids.append(list(old_centroids[k]))
        return centroids

    @staticmethod
    def _centroid_shift(
        old: List[List[float]],
        new: List[List[float]],
    ) -> float:
        """Maximum Euclidean shift among all centroids."""
        return max(
            euclidean_distance(o, n) for o, n in zip(old, new)
        )

    @staticmethod
    def _compute_inertia(
        X: List[List[float]],
        labels: List[int],
        centroids: List[List[float]],
    ) -> float:
        """Sum of squared distances to nearest centroid."""
        inertia = 0.0
        for i, x in enumerate(X):
            c = centroids[labels[i]]
            inertia += sum(
                (xi - ci) ** 2 for xi, ci in zip(x, c)
            )
        return inertia

    @staticmethod
    def _silhouette(
        X: List[List[float]],
        labels: List[int],
    ) -> float:
        """Compute mean silhouette coefficient.

        For each sample i:
            a(i) = mean intra-cluster distance
            b(i) = mean nearest-cluster distance
            s(i) = (b(i) - a(i)) / max(a(i), b(i))

        Parameters
        ----------
        X : List[List[float]]
            Data matrix.
        labels : List[int]
            Cluster assignments.

        Returns
        -------
        float
            Mean silhouette score in [-1, 1].
        """
        n = len(X)
        if n <= 1:
            return 0.0

        unique_labels = set(labels)
        if len(unique_labels) <= 1 or len(unique_labels) >= n:
            return 0.0

        # Group indices by cluster
        clusters: Dict[int, List[int]] = {}
        for i, lab in enumerate(labels):
            clusters.setdefault(lab, []).append(i)

        total_sil = 0.0
        for i in range(n):
            lab_i = labels[i]
            cluster_i = clusters[lab_i]

            # a(i): mean distance to same-cluster points
            if len(cluster_i) <= 1:
                a_i = 0.0
            else:
                a_i = sum(
                    euclidean_distance(X[i], X[j])
                    for j in cluster_i if j != i
                ) / (len(cluster_i) - 1)

            # b(i): min over other clusters of mean distance
            b_i = math.inf
            for lab_other, members in clusters.items():
                if lab_other == lab_i:
                    continue
                mean_d = sum(
                    euclidean_distance(X[i], X[j])
                    for j in members
                ) / len(members)
                if mean_d < b_i:
                    b_i = mean_d

            denom = max(a_i, b_i)
            if denom > 0:
                total_sil += (b_i - a_i) / denom
            # else s(i) = 0, already handled

        return total_sil / n
