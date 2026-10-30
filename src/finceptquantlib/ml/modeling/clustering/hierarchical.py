"""
Agglomerative Hierarchical Clustering

Bottom-up hierarchical clustering with single, complete, average,
and Ward linkage. Produces a full linkage matrix and dendrogram
ordering. All operations use pure Python with no NumPy dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

from finceptquantlib.ml._base import Clusterer
from finceptquantlib.ml._matrix_utils import (
    distance_matrix,
    euclidean_distance,
)


class LinkageMethod(Enum):
    """Linkage strategy for agglomerative clustering.

    Attributes
    ----------
    SINGLE : str
        Minimum pairwise distance between clusters.
    COMPLETE : str
        Maximum pairwise distance between clusters.
    AVERAGE : str
        Mean pairwise distance between clusters.
    WARD : str
        Minimum increase in total within-cluster variance.
    """

    SINGLE = "single"
    COMPLETE = "complete"
    AVERAGE = "average"
    WARD = "ward"


@dataclass
class HierarchicalResult:
    """Result of agglomerative hierarchical clustering.

    Attributes
    ----------
    labels : List[int]
        Cluster assignment for each sample after cutting
        the dendrogram at ``n_clusters``.
    linkage_matrix : List[List[float]]
        Linkage matrix with rows ``[cluster_i, cluster_j,
        distance, size]`` recording each merge step.
    n_clusters : int
        Number of clusters in the final partition.
    dendrogram_order : List[int]
        Leaf ordering for dendrogram visualization.
    """

    labels: List[int]
    linkage_matrix: List[List[float]]
    n_clusters: int
    dendrogram_order: List[int]


class AgglomerativeClustering(Clusterer):
    """Agglomerative (bottom-up) hierarchical clustering.

    Parameters
    ----------
    n_clusters : int
        Desired number of clusters after cutting the dendrogram.
    linkage : str
        Linkage criterion: ``"single"``, ``"complete"``,
        ``"average"``, or ``"ward"``.
    metric : str
        Distance metric. Currently only ``"euclidean"`` is
        supported.

    Examples
    --------
    >>> from finceptquantlib.ml.modeling.clustering import (
    ...     AgglomerativeClustering,
    ... )
    >>> X = [[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]]
    >>> hc = AgglomerativeClustering(n_clusters=2, linkage="ward")
    >>> result = hc.fit(X)
    >>> result.n_clusters
    2
    """

    def __init__(
        self,
        n_clusters: int = 3,
        linkage: str = "ward",
        metric: str = "euclidean",
    ) -> None:
        self.n_clusters = n_clusters
        self.linkage = LinkageMethod(linkage)
        self.metric = metric
        self._linkage_matrix: List[List[float]] = []
        self._cluster_centers: List[List[float]] = []
        self._labels: List[int] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: List[List[float]]) -> HierarchicalResult:
        """Fit agglomerative clustering to data.

        Parameters
        ----------
        X : List[List[float]]
            Data matrix (n_samples x n_features).

        Returns
        -------
        HierarchicalResult
            Clustering result with labels, linkage matrix,
            and dendrogram order.
        """
        dist = distance_matrix(X)
        return self._fit_core(X, dist)

    def fit_from_distance(
        self, dist_matrix: List[List[float]]
    ) -> HierarchicalResult:
        """Fit from a precomputed distance matrix.

        Parameters
        ----------
        dist_matrix : List[List[float]]
            Precomputed pairwise distance matrix.

        Returns
        -------
        HierarchicalResult
            Clustering result.

        Raises
        ------
        ValueError
            If linkage is ``"ward"`` without original data.
        """
        n = len(dist_matrix)
        # For ward we need raw data; fabricate simple 1-D embedding
        # by using classical MDS (not ideal but allows the method).
        if self.linkage == LinkageMethod.WARD:
            raise ValueError(
                "Ward linkage requires raw data. Use fit() instead."
            )
        dummy_X = [[0.0] for _ in range(n)]
        return self._fit_core(dummy_X, dist_matrix)

    def predict(self, X: List[List[float]]) -> List[int]:
        """Assign new points to nearest cluster center.

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
        if not self._cluster_centers:
            raise RuntimeError("Model has not been fit yet.")
        labels: List[int] = []
        for x in X:
            best_k = 0
            best_d = math.inf
            for k, center in enumerate(self._cluster_centers):
                d = euclidean_distance(x, center)
                if d < best_d:
                    best_d = d
                    best_k = k
            labels.append(best_k)
        return labels

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_core(
        self,
        X: List[List[float]],
        dist: List[List[float]],
    ) -> HierarchicalResult:
        """Core agglomerative algorithm.

        Parameters
        ----------
        X : List[List[float]]
            Original data (needed for Ward linkage center
            computation).
        dist : List[List[float]]
            Pairwise distance matrix.

        Returns
        -------
        HierarchicalResult
            Full clustering result.
        """
        n = len(dist)
        m = len(X[0]) if X else 0

        # Active clusters: id -> list of original sample indices
        clusters: Dict[int, List[int]] = {
            i: [i] for i in range(n)
        }
        # Cluster centers for Ward linkage
        centers: Dict[int, List[float]] = {
            i: list(X[i]) for i in range(n)
        }

        # Work on a mutable copy of the distance matrix stored
        # as dict of dicts for active clusters only.
        active_dist: Dict[int, Dict[int, float]] = {}
        for i in range(n):
            active_dist[i] = {}
            for j in range(n):
                if i != j:
                    active_dist[i][j] = dist[i][j]

        linkage_mat: List[List[float]] = []
        next_id = n  # new cluster ids start at n

        for step in range(n - 1):
            # Find closest pair among active clusters
            best_i = -1
            best_j = -1
            best_d = math.inf
            active_ids = sorted(active_dist.keys())

            for idx_a in range(len(active_ids)):
                ci = active_ids[idx_a]
                for idx_b in range(idx_a + 1, len(active_ids)):
                    cj = active_ids[idx_b]
                    d = active_dist[ci][cj]
                    if d < best_d:
                        best_d = d
                        best_i = ci
                        best_j = cj

            # Merge best_i and best_j into next_id
            merged = clusters[best_i] + clusters[best_j]
            new_size = len(merged)
            linkage_mat.append(
                [float(best_i), float(best_j), best_d, float(new_size)]
            )

            # Compute new center for Ward
            ni = len(clusters[best_i])
            nj = len(clusters[best_j])
            new_center = [
                (centers[best_i][f] * ni + centers[best_j][f] * nj)
                / new_size
                for f in range(m)
            ]

            # Compute distances from new cluster to all remaining
            new_dists: Dict[int, float] = {}
            for ck in active_ids:
                if ck == best_i or ck == best_j:
                    continue
                nk = len(clusters[ck])

                if self.linkage == LinkageMethod.SINGLE:
                    d_new = min(
                        active_dist[best_i][ck],
                        active_dist[best_j][ck],
                    )
                elif self.linkage == LinkageMethod.COMPLETE:
                    d_new = max(
                        active_dist[best_i][ck],
                        active_dist[best_j][ck],
                    )
                elif self.linkage == LinkageMethod.AVERAGE:
                    d_new = (
                        active_dist[best_i][ck] * ni
                        + active_dist[best_j][ck] * nj
                    ) / new_size
                elif self.linkage == LinkageMethod.WARD:
                    # Lance-Williams recurrence for Ward
                    d_ik = active_dist[best_i][ck] ** 2
                    d_jk = active_dist[best_j][ck] ** 2
                    d_ij = best_d ** 2
                    ntot = ni + nj + nk
                    d_new = math.sqrt(
                        max(
                            0.0,
                            ((ni + nk) * d_ik
                             + (nj + nk) * d_jk
                             - nk * d_ij) / ntot,
                        )
                    )
                else:
                    d_new = active_dist[best_i][ck]

                new_dists[ck] = d_new

            # Remove old clusters from active_dist
            del active_dist[best_i]
            del active_dist[best_j]
            for ck_dists in active_dist.values():
                ck_dists.pop(best_i, None)
                ck_dists.pop(best_j, None)

            # Add new cluster
            active_dist[next_id] = new_dists
            for ck, d_val in new_dists.items():
                active_dist[ck][next_id] = d_val

            clusters[next_id] = merged
            centers[next_id] = new_center
            del clusters[best_i]
            del clusters[best_j]
            del centers[best_i]
            del centers[best_j]

            next_id += 1

        # Cut dendrogram at n_clusters
        labels = self._cut_dendrogram(n, linkage_mat, self.n_clusters)
        dendrogram_order = self._dendrogram_order(n, linkage_mat)

        # Store cluster centers for predict
        cluster_ids = set(labels)
        self._cluster_centers = []
        label_map: Dict[int, int] = {}
        for new_lab, cid in enumerate(sorted(cluster_ids)):
            label_map[cid] = new_lab
            members = [
                X[i] for i in range(n) if labels[i] == cid
            ]
            center = [0.0] * m
            for pt in members:
                for f in range(m):
                    center[f] += pt[f]
            if members:
                center = [c / len(members) for c in center]
            self._cluster_centers.append(center)

        # Remap labels to 0..n_clusters-1
        labels = [label_map[lab] for lab in labels]
        self._labels = labels
        self._linkage_matrix = linkage_mat

        return HierarchicalResult(
            labels=labels,
            linkage_matrix=linkage_mat,
            n_clusters=self.n_clusters,
            dendrogram_order=dendrogram_order,
        )

    @staticmethod
    def _cut_dendrogram(
        n: int,
        linkage_mat: List[List[float]],
        n_clusters: int,
    ) -> List[int]:
        """Cut dendrogram to produce n_clusters clusters.

        Parameters
        ----------
        n : int
            Number of original samples.
        linkage_mat : List[List[float]]
            Full linkage matrix.
        n_clusters : int
            Desired number of clusters.

        Returns
        -------
        List[int]
            Cluster label for each original sample.
        """
        # Number of merges to perform = n - n_clusters
        merges_to_use = n - n_clusters

        # Build cluster membership by replaying merges
        membership: Dict[int, List[int]] = {
            i: [i] for i in range(n)
        }
        for step in range(merges_to_use):
            row = linkage_mat[step]
            ci = int(row[0])
            cj = int(row[1])
            new_id = n + step
            membership[new_id] = membership[ci] + membership[cj]
            del membership[ci]
            del membership[cj]

        labels = [0] * n
        for lab, (cid, members) in enumerate(
            sorted(membership.items())
        ):
            for idx in members:
                labels[idx] = lab
        return labels

    @staticmethod
    def _dendrogram_order(
        n: int,
        linkage_mat: List[List[float]],
    ) -> List[int]:
        """Compute leaf ordering for dendrogram display.

        Uses a simple in-order traversal of the merge tree.

        Parameters
        ----------
        n : int
            Number of original samples.
        linkage_mat : List[List[float]]
            Full linkage matrix.

        Returns
        -------
        List[int]
            Ordered list of original sample indices.
        """
        # Build tree: each merge creates a node
        children: Dict[int, List[int]] = {}
        for step, row in enumerate(linkage_mat):
            node_id = n + step
            children[node_id] = [int(row[0]), int(row[1])]

        # In-order traversal (iterative to avoid recursion limits)
        root = n + len(linkage_mat) - 1
        order: List[int] = []
        stack: List[int] = [root]
        while stack:
            node = stack.pop()
            if node < n:
                order.append(node)
            else:
                left, right = children[node]
                # Push right first so left is processed first
                stack.append(right)
                stack.append(left)
        return order
