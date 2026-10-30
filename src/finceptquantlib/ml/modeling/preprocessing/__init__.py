"""
Preprocessing Subpackage

Feature preprocessing transformers for financial time-series and
cross-sectional data. All transformers implement the ``Transformer``
ABC and operate on ``List[List[float]]`` matrices using pure Python.

Modules
-------
scalers
    ZScoreScaler, MinMaxScaler, RobustScaler
winsorizer
    WinsorizeMethod, Winsorizer
missing
    MissingValueMethod, MissingValueHandler
outliers
    OutlierMethod, OutlierResult, OutlierDetector
stationarity
    StationarityTestResult, log_returns, simple_returns,
    difference, fractional_difference, adf_test_statistic
transforms
    BoxCoxTransformer, PowerTransformer
"""

from finceptquantlib.ml.modeling.preprocessing.scalers import (
    MinMaxScaler,
    RobustScaler,
    ZScoreScaler,
)
from finceptquantlib.ml.modeling.preprocessing.winsorizer import (
    WinsorizeMethod,
    Winsorizer,
)
from finceptquantlib.ml.modeling.preprocessing.missing import (
    MissingValueHandler,
    MissingValueMethod,
)
from finceptquantlib.ml.modeling.preprocessing.outliers import (
    OutlierDetector,
    OutlierMethod,
    OutlierResult,
)
from finceptquantlib.ml.modeling.preprocessing.stationarity import (
    StationarityTestResult,
    adf_test_statistic,
    difference,
    fractional_difference,
    log_returns,
    simple_returns,
)
from finceptquantlib.ml.modeling.preprocessing.transforms import (
    BoxCoxTransformer,
    PowerTransformer,
)

__all__ = [
    # scalers
    "ZScoreScaler",
    "MinMaxScaler",
    "RobustScaler",
    # winsorizer
    "WinsorizeMethod",
    "Winsorizer",
    # missing
    "MissingValueMethod",
    "MissingValueHandler",
    # outliers
    "OutlierMethod",
    "OutlierResult",
    "OutlierDetector",
    # stationarity
    "StationarityTestResult",
    "log_returns",
    "simple_returns",
    "difference",
    "fractional_difference",
    "adf_test_statistic",
    # transforms
    "BoxCoxTransformer",
    "PowerTransformer",
]
