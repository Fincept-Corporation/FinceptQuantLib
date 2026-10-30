"""
ML Modeling Module

General-purpose machine learning modeling:
- Preprocessing: Scaling, missing values, outliers
- Clustering: KMeans, hierarchical, DBSCAN, isolation forest
- Regression: Linear models, trees, gradient boosting
- Feature Engineering: Technical indicators, rolling stats
"""

from .preprocessing import *
from .clustering import *
from .regression import *
from .feature_engineering import *

# Combine __all__ from all submodules
from .preprocessing import __all__ as _preprocessing_all
from .clustering import __all__ as _clustering_all
from .regression import __all__ as _regression_all
from .feature_engineering import __all__ as _feature_engineering_all

__all__ = []
__all__.extend(_preprocessing_all)
__all__.extend(_clustering_all)
__all__.extend(_regression_all)
__all__.extend(_feature_engineering_all)
