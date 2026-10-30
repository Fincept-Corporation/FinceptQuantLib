"""
ML-Based Curve and Volatility Surface Fitting

Provides machine learning alternatives for yield curve interpolation
and volatility surface fitting. These models implement the same
method interfaces as the analytical counterparts in
``finceptquantlib.curves`` and ``finceptquantlib.volatility``.

Classes
-------
GaussianProcessCurve
    GP regression for yield curve fitting with uncertainty.
GPCurveResult
    Dataclass result from GP curve fitting.
KernelType
    Enum of supported GP kernels.

NeuralNetCurve
    Feedforward NN for yield curve fitting.
NNCurveResult
    Dataclass result from NN curve fitting.

CurveAutoencoder
    Autoencoder for curve dimensionality reduction.
AutoencoderResult
    Dataclass result from autoencoder training.

GPVolSurface
    2D GP regression for volatility surfaces.
GPVolSurfaceResult
    Dataclass result from GP vol surface fitting.

NeuralNetVolSurface
    Feedforward NN for volatility surfaces.
NNVolSurfaceResult
    Dataclass result from NN vol surface fitting.
"""

from finceptquantlib.ml.markets.curves_ml.gp_curve import (
    GaussianProcessCurve,
    GPCurveResult,
    KernelType,
)
from finceptquantlib.ml.markets.curves_ml.nn_curve import (
    NeuralNetCurve,
    NNCurveResult,
)
from finceptquantlib.ml.markets.curves_ml.autoencoder_curve import (
    CurveAutoencoder,
    AutoencoderResult,
)
from finceptquantlib.ml.markets.curves_ml.gp_vol_surface import (
    GPVolSurface,
    GPVolSurfaceResult,
)
from finceptquantlib.ml.markets.curves_ml.nn_vol_surface import (
    NeuralNetVolSurface,
    NNVolSurfaceResult,
)

__all__ = [
    "GaussianProcessCurve",
    "GPCurveResult",
    "KernelType",
    "NeuralNetCurve",
    "NNCurveResult",
    "CurveAutoencoder",
    "AutoencoderResult",
    "GPVolSurface",
    "GPVolSurfaceResult",
    "NeuralNetVolSurface",
    "NNVolSurfaceResult",
]
