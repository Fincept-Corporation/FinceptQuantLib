"""
Clustering and Dimensionality Reduction Subpackage

Provides unsupervised learning algorithms for grouping data
points, detecting anomalies, and reducing dimensionality.
All implementations are pure Python with no NumPy dependency.

Classes
-------
KMeans
    K-Means clustering with K-Means++ initialization.
KMeansResult
    Result dataclass for K-Means.
AgglomerativeClustering
    Bottom-up hierarchical clustering.
HierarchicalResult
    Result dataclass for hierarchical clustering.
LinkageMethod
    Enum of linkage strategies.
PCA
    Principal Component Analysis via power iteration.
PCAResult
    Result dataclass for PCA.
IsolationForest
    Isolation Forest anomaly detection.
IsolationForestResult
    Result dataclass for Isolation Forest.
DBSCAN
    Density-based spatial clustering.
DBSCANResult
    Result dataclass for DBSCAN.
"""

from finceptquantlib.ml.modeling.clustering.dbscan import (
    DBSCAN,
    DBSCANResult,
)
from finceptquantlib.ml.modeling.clustering.hierarchical import (
    AgglomerativeClustering,
    HierarchicalResult,
    LinkageMethod,
)
from finceptquantlib.ml.modeling.clustering.isolation_forest import (
    IsolationForest,
    IsolationForestResult,
)
from finceptquantlib.ml.modeling.clustering.kmeans import (
    KMeans,
    KMeansResult,
)
from finceptquantlib.ml.modeling.clustering.pca import (
    PCA,
    PCAResult,
)

__all__ = [
    "DBSCAN",
    "DBSCANResult",
    "AgglomerativeClustering",
    "HierarchicalResult",
    "IsolationForest",
    "IsolationForestResult",
    "KMeans",
    "KMeansResult",
    "LinkageMethod",
    "PCA",
    "PCAResult",
]
