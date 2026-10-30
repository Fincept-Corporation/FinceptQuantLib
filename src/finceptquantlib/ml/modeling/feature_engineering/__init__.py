"""
Feature Engineering Subpackage

Provides feature engineering tools for financial machine learning:

- **technical**: Technical analysis indicators (SMA, EMA, RSI, MACD, etc.)
- **rolling**: Rolling and expanding window statistics
- **cross_sectional**: Cross-sectional transformations (rank, zscore, etc.)
- **calendar_features**: Date-based features for temporal patterns
- **lags**: Lag, lead, and return feature generators
- **financial_ratios**: ML-friendly wrappers for financial ratio computation
- **pipeline**: Composable feature transformation pipeline
"""

from finceptquantlib.ml.modeling.feature_engineering.technical import (
    sma,
    ema,
    rsi,
    macd,
    bollinger_bands,
    atr,
    obv,
    vwap,
    stochastic_oscillator,
    adx,
    williams_r,
    cci,
)

from finceptquantlib.ml.modeling.feature_engineering.rolling import (
    rolling_mean,
    rolling_std,
    rolling_skewness,
    rolling_kurtosis,
    rolling_zscore,
    rolling_correlation,
    rolling_beta,
    expanding_mean,
    expanding_std,
)

from finceptquantlib.ml.modeling.feature_engineering.cross_sectional import (
    cross_sectional_rank,
    cross_sectional_percentile,
    cross_sectional_zscore,
    cross_sectional_demean,
    winsorized_zscore,
)

from finceptquantlib.ml.modeling.feature_engineering.calendar_features import (
    month_of_year,
    quarter,
    day_of_week,
    day_of_month,
    is_month_end,
    is_quarter_end,
    is_year_end,
    days_to_expiry,
    trading_day_of_month,
)

from finceptquantlib.ml.modeling.feature_engineering.lags import (
    add_lags,
    add_leads,
    add_returns,
    add_log_returns,
)

from finceptquantlib.ml.modeling.feature_engineering.financial_ratios import (
    compute_ratio_features,
)

from finceptquantlib.ml.modeling.feature_engineering.pipeline import (
    FeaturePipeline,
)

__all__ = [
    # Technical indicators
    "sma",
    "ema",
    "rsi",
    "macd",
    "bollinger_bands",
    "atr",
    "obv",
    "vwap",
    "stochastic_oscillator",
    "adx",
    "williams_r",
    "cci",
    # Rolling statistics
    "rolling_mean",
    "rolling_std",
    "rolling_skewness",
    "rolling_kurtosis",
    "rolling_zscore",
    "rolling_correlation",
    "rolling_beta",
    "expanding_mean",
    "expanding_std",
    # Cross-sectional
    "cross_sectional_rank",
    "cross_sectional_percentile",
    "cross_sectional_zscore",
    "cross_sectional_demean",
    "winsorized_zscore",
    # Calendar features
    "month_of_year",
    "quarter",
    "day_of_week",
    "day_of_month",
    "is_month_end",
    "is_quarter_end",
    "is_year_end",
    "days_to_expiry",
    "trading_day_of_month",
    # Lags and returns
    "add_lags",
    "add_leads",
    "add_returns",
    "add_log_returns",
    # Financial ratios
    "compute_ratio_features",
    # Pipeline
    "FeaturePipeline",
]
