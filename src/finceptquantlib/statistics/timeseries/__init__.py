"""
Time Series Analysis

Models for time series analysis:
- ARIMA (AR, MA, ARMA, ARIMA)
- GARCH family (GARCH, EGARCH, GJR-GARCH)
"""

from finceptquantlib.statistics.timeseries.arima import (
    AR,
    MA,
    ARIMA,
    ARIMAResult,
)

from finceptquantlib.statistics.timeseries.garch import (
    GARCH,
    EGARCH,
    GJRGARCH,
    GARCHResult,
)

__all__ = [
    # ARIMA
    "AR",
    "MA",
    "ARIMA",
    "ARIMAResult",
    # GARCH
    "GARCH",
    "EGARCH",
    "GJRGARCH",
    "GARCHResult",
]
