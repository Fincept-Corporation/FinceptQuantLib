"""Tests for finceptquantlib.ml.modeling.clustering module."""

import pytest

from finceptquantlib.ml.modeling.clustering import (
    KMeans,
    KMeansResult,
    AgglomerativeClustering,
    HierarchicalResult,
    LinkageMethod,
    PCA,
    PCAResult,
    IsolationForest,
    IsolationForestResult,
    DBSCAN,
    DBSCANResult,
)


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

TWO_CLUSTER_DATA = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [10.0, 10.0],
    [10.0, 11.0],
    [11.0, 10.0],
    [11.0, 11.0],
]


# ---------------------------------------------------------------------------
# KMeans
# ---------------------------------------------------------------------------

class TestKMeans:
    """Tests for KMeans clustering."""

    def test_fit_returns_kmeans_result(self):
        km = KMeans(n_clusters=2)
        result = km.fit(TWO_CLUSTER_DATA)
        assert isinstance(result, KMeansResult)

    def test_result_has_required_attributes(self):
        km = KMeans(n_clusters=2)
        result = km.fit(TWO_CLUSTER_DATA)
        assert hasattr(result, "labels")
        assert hasattr(result, "centroids")
        assert hasattr(result, "inertia")
        assert hasattr(result, "n_iterations")

    def test_correct_number_of_labels(self):
        km = KMeans(n_clusters=2)
        result = km.fit(TWO_CLUSTER_DATA)
        assert len(result.labels) == len(TWO_CLUSTER_DATA)

    def test_two_clusters_found(self):
        km = KMeans(n_clusters=2)
        result = km.fit(TWO_CLUSTER_DATA)
        unique_labels = set(result.labels)
        assert len(unique_labels) == 2

    def test_clusters_separate_groups(self):
        """Points near (0,0) should be in one cluster, points near (10,10) in another."""
        km = KMeans(n_clusters=2)
        result = km.fit(TWO_CLUSTER_DATA)

        # First 4 points should share a label, last 4 should share a different label
        low_labels = set(result.labels[:4])
        high_labels = set(result.labels[4:])
        assert len(low_labels) == 1, "All low-value points should be in one cluster"
        assert len(high_labels) == 1, "All high-value points should be in one cluster"
        assert low_labels != high_labels, "The two groups should have different labels"

    def test_correct_number_of_centers(self):
        km = KMeans(n_clusters=2)
        result = km.fit(TWO_CLUSTER_DATA)
        assert len(result.centroids) == 2

    def test_inertia_is_non_negative(self):
        km = KMeans(n_clusters=2)
        result = km.fit(TWO_CLUSTER_DATA)
        assert result.inertia >= 0.0

    def test_iterations_is_positive(self):
        km = KMeans(n_clusters=2)
        result = km.fit(TWO_CLUSTER_DATA)
        assert result.n_iterations > 0

    def test_predict_assigns_labels(self):
        km = KMeans(n_clusters=2)
        km.fit(TWO_CLUSTER_DATA)
        new_points = [[0.5, 0.5], [10.5, 10.5]]
        predicted = km.predict(new_points)
        assert len(predicted) == 2
        # The two test points should get different labels
        assert predicted[0] != predicted[1]

    @pytest.mark.slow
    def test_larger_dataset(self):
        """Test KMeans with a larger generated dataset."""
        import random

        random.seed(42)
        data = []
        for _ in range(100):
            data.append([random.gauss(0, 1), random.gauss(0, 1)])
        for _ in range(100):
            data.append([random.gauss(10, 1), random.gauss(10, 1)])

        km = KMeans(n_clusters=2)
        result = km.fit(data)
        assert len(set(result.labels)) == 2


# ---------------------------------------------------------------------------
# AgglomerativeClustering
# ---------------------------------------------------------------------------

class TestAgglomerativeClustering:
    """Tests for AgglomerativeClustering."""

    def test_fit_returns_hierarchical_result(self):
        ac = AgglomerativeClustering(n_clusters=2, linkage=LinkageMethod.SINGLE)
        result = ac.fit(TWO_CLUSTER_DATA)
        assert isinstance(result, HierarchicalResult)

    def test_result_has_labels(self):
        ac = AgglomerativeClustering(n_clusters=2, linkage=LinkageMethod.SINGLE)
        result = ac.fit(TWO_CLUSTER_DATA)
        assert hasattr(result, "labels")
        assert len(result.labels) == len(TWO_CLUSTER_DATA)

    def test_two_clusters_found(self):
        ac = AgglomerativeClustering(n_clusters=2, linkage=LinkageMethod.SINGLE)
        result = ac.fit(TWO_CLUSTER_DATA)
        unique_labels = set(result.labels)
        assert len(unique_labels) == 2

    def test_clusters_separate_groups(self):
        ac = AgglomerativeClustering(n_clusters=2, linkage=LinkageMethod.SINGLE)
        result = ac.fit(TWO_CLUSTER_DATA)

        low_labels = set(result.labels[:4])
        high_labels = set(result.labels[4:])
        assert len(low_labels) == 1
        assert len(high_labels) == 1
        assert low_labels != high_labels


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

class TestPCA:
    """Tests for PCA (Principal Component Analysis)."""

    def test_fit_returns_pca_result(self):
        pca = PCA(n_components=1)
        result = pca.fit(TWO_CLUSTER_DATA)
        assert isinstance(result, PCAResult)

    def test_result_has_required_attributes(self):
        pca = PCA(n_components=1)
        result = pca.fit(TWO_CLUSTER_DATA)
        assert hasattr(result, "components")
        assert hasattr(result, "explained_variance")
        assert hasattr(result, "explained_variance_ratio")

    def test_transform_reduces_dimensionality(self):
        pca = PCA(n_components=1)
        pca.fit(TWO_CLUSTER_DATA)
        transformed = pca.transform(TWO_CLUSTER_DATA)

        assert len(transformed) == len(TWO_CLUSTER_DATA)
        # Each row should have n_components features
        assert len(transformed[0]) == 1

    def test_n_components_less_than_original(self):
        original_features = len(TWO_CLUSTER_DATA[0])
        n_components = 1
        assert n_components < original_features

        pca = PCA(n_components=n_components)
        pca.fit(TWO_CLUSTER_DATA)
        transformed = pca.transform(TWO_CLUSTER_DATA)
        assert len(transformed[0]) == n_components

    def test_explained_variance_ratios_sum_to_one(self):
        """When using all components, explained variance ratios should sum to ~1."""
        n_features = len(TWO_CLUSTER_DATA[0])
        pca = PCA(n_components=n_features)
        result = pca.fit(TWO_CLUSTER_DATA)

        total_ratio = sum(result.explained_variance_ratio)
        assert total_ratio == pytest.approx(1.0, abs=1e-6)

    def test_explained_variance_is_non_negative(self):
        pca = PCA(n_components=2)
        result = pca.fit(TWO_CLUSTER_DATA)

        for var in result.explained_variance:
            assert var >= 0.0

    def test_explained_variance_is_sorted_descending(self):
        pca = PCA(n_components=2)
        result = pca.fit(TWO_CLUSTER_DATA)

        for i in range(len(result.explained_variance) - 1):
            assert result.explained_variance[i] >= result.explained_variance[i + 1]


# ---------------------------------------------------------------------------
# IsolationForest
# ---------------------------------------------------------------------------

class TestIsolationForest:
    """Tests for IsolationForest anomaly detection."""

    def test_fit_returns_isolation_forest_result(self):
        iforest = IsolationForest(n_estimators=50, contamination=0.1)
        result = iforest.fit(TWO_CLUSTER_DATA)
        assert isinstance(result, IsolationForestResult)

    def test_scores_for_all_points(self):
        iforest = IsolationForest(n_estimators=50, contamination=0.1)
        result = iforest.fit(TWO_CLUSTER_DATA)
        assert hasattr(result, "anomaly_scores")
        assert len(result.anomaly_scores) == len(TWO_CLUSTER_DATA)

    def test_outlier_has_higher_score(self):
        """An extreme outlier should receive a different (higher anomaly) score."""
        normal_data = [[float(i), float(i)] for i in range(20)]
        data_with_outlier = normal_data + [[1000.0, 1000.0]]

        iforest = IsolationForest(n_estimators=50, contamination=0.1)
        result = iforest.fit(data_with_outlier)

        # The outlier (last point) should have the highest anomaly score
        outlier_score = result.anomaly_scores[-1]
        normal_scores = result.anomaly_scores[:-1]
        max_normal_score = max(normal_scores)
        assert outlier_score >= max_normal_score

    @pytest.mark.slow
    def test_with_many_trees(self):
        """Test with a larger number of trees for stability."""
        normal_data = [[float(i), float(i)] for i in range(50)]
        data_with_outlier = normal_data + [[500.0, 500.0]]

        iforest = IsolationForest(n_estimators=200, contamination=0.05)
        result = iforest.fit(data_with_outlier)
        assert len(result.anomaly_scores) == len(data_with_outlier)


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------

class TestDBSCAN:
    """Tests for DBSCAN density-based clustering."""

    def test_fit_returns_dbscan_result(self):
        dbscan = DBSCAN(eps=1.5, min_samples=2)
        result = dbscan.fit(TWO_CLUSTER_DATA)
        assert isinstance(result, DBSCANResult)

    def test_result_has_labels(self):
        dbscan = DBSCAN(eps=1.5, min_samples=2)
        result = dbscan.fit(TWO_CLUSTER_DATA)
        assert hasattr(result, "labels")
        assert len(result.labels) == len(TWO_CLUSTER_DATA)

    def test_two_clusters_with_separation(self):
        """Well-separated data with eps=1.5 should produce two clusters."""
        dbscan = DBSCAN(eps=1.5, min_samples=2)
        result = dbscan.fit(TWO_CLUSTER_DATA)

        # Filter out noise labels (-1)
        non_noise_labels = [l for l in result.labels if l != -1]
        unique_clusters = set(non_noise_labels)
        assert len(unique_clusters) == 2

    def test_noise_label_is_negative_one(self):
        """Points that don't meet density requirements should get label -1."""
        # Add an isolated point far from both clusters
        data_with_noise = TWO_CLUSTER_DATA + [[50.0, 50.0]]
        dbscan = DBSCAN(eps=1.5, min_samples=2)
        result = dbscan.fit(data_with_noise)

        # The isolated point should be labelled as noise
        assert result.labels[-1] == -1

    def test_all_noise_with_strict_params(self):
        """With very small eps, most or all points should be noise."""
        dbscan = DBSCAN(eps=0.01, min_samples=5)
        result = dbscan.fit(TWO_CLUSTER_DATA)

        noise_count = sum(1 for l in result.labels if l == -1)
        assert noise_count == len(TWO_CLUSTER_DATA)

    def test_single_cluster_with_large_eps(self):
        """With a very large eps, all points should be in one cluster."""
        dbscan = DBSCAN(eps=100.0, min_samples=2)
        result = dbscan.fit(TWO_CLUSTER_DATA)

        non_noise_labels = [l for l in result.labels if l != -1]
        unique_clusters = set(non_noise_labels)
        assert len(unique_clusters) == 1
