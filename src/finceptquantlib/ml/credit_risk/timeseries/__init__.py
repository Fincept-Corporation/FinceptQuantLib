"""
Time Series ML Subpackage

Provides regime detection, changepoint detection, ML-augmented GARCH
volatility modelling, feature importance analysis, and time-series-
aware cross-validation strategies.

Classes
-------
GaussianHMM
    Gaussian Hidden Markov Model for regime detection.
CUSUM
    Cumulative Sum changepoint detector.
BayesianChangepoint
    Bayesian online changepoint detection (Adams & MacKay 2007).
MLGARCHHybrid
    ML-augmented GARCH(p,q) conditional volatility model.
WalkForwardCV
    Walk-forward (rolling window) cross-validation.
ExpandingWindowCV
    Expanding window cross-validation.
PurgedKFoldCV
    Purged K-Fold cross-validation (Lopez de Prado).

Dataclasses
-----------
HMMResult
    Result of GaussianHMM fitting.
ChangepointResult
    Result of changepoint detection.
MLGARCHResult
    Result of ML-augmented GARCH fitting.
TimeSeriesImportanceResult
    Result of feature importance analysis.
CVResult
    Result of cross-validation evaluation.

Functions
---------
time_series_feature_importance
    Permutation-based feature importance for time series.
sequential_feature_selection
    Sequential (forward/backward) feature selection.
"""

from finceptquantlib.ml.credit_risk.timeseries.regime import (
    GaussianHMM,
    HMMResult,
)
from finceptquantlib.ml.credit_risk.timeseries.changepoint import (
    BayesianChangepoint,
    ChangepointResult,
    CUSUM,
)
from finceptquantlib.ml.credit_risk.timeseries.ml_garch import (
    MLGARCHHybrid,
    MLGARCHResult,
)
from finceptquantlib.ml.credit_risk.timeseries.importance import (
    TimeSeriesImportanceResult,
    sequential_feature_selection,
    time_series_feature_importance,
)
from finceptquantlib.ml.credit_risk.timeseries.cross_validation import (
    CVResult,
    ExpandingWindowCV,
    PurgedKFoldCV,
    WalkForwardCV,
)

__all__ = [
    # Regime detection
    "GaussianHMM",
    "HMMResult",
    # Changepoint detection
    "CUSUM",
    "BayesianChangepoint",
    "ChangepointResult",
    # GARCH
    "MLGARCHHybrid",
    "MLGARCHResult",
    # Feature importance
    "TimeSeriesImportanceResult",
    "time_series_feature_importance",
    "sequential_feature_selection",
    # Cross-validation
    "CVResult",
    "WalkForwardCV",
    "ExpandingWindowCV",
    "PurgedKFoldCV",
]
