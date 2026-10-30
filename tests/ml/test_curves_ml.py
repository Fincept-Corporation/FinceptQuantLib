"""Tests for ML-based curve fitting and volatility surface models."""

from datetime import date

import pytest

from finceptquantlib.ml.markets.curves_ml import (
    CurveAutoencoder,
    AutoencoderResult,
    GaussianProcessCurve,
    GPCurveResult,
    GPVolSurface,
    GPVolSurfaceResult,
    NeuralNetCurve,
    NNCurveResult,
    NeuralNetVolSurface,
    NNVolSurfaceResult,
)


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

REF_DATE = date(2024, 1, 1)
MATURITIES = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
RATES = [4.5, 4.3, 4.0, 3.8, 3.5, 3.2]


# ---------------------------------------------------------------------------
# GaussianProcessCurve
# ---------------------------------------------------------------------------


class TestGaussianProcessCurve:
    """Tests for the Gaussian Process yield-curve fitter."""

    @pytest.mark.slow
    def test_constructor_defaults(self):
        gp = GaussianProcessCurve(reference_date=REF_DATE, kernel="rbf", length_scale=1.0, noise_variance=0.01)
        assert gp is not None

    @pytest.mark.slow
    def test_fit_returns_gpcurve_result(self):
        gp = GaussianProcessCurve(reference_date=REF_DATE, kernel="rbf", length_scale=1.0, noise_variance=0.01)
        result = gp.fit(MATURITIES, RATES)

        assert isinstance(result, GPCurveResult)
        assert result.mean is not None
        assert result.std is not None
        assert result.log_marginal_likelihood is not None

    @pytest.mark.slow
    def test_zero_rate_returns_float(self):
        gp = GaussianProcessCurve(reference_date=REF_DATE, kernel="rbf", length_scale=1.0, noise_variance=0.01)
        gp.fit(MATURITIES, RATES)

        for t in [0.25, 1.0, 3.0, 7.0, 10.0]:
            rate = gp.zero_rate(t)
            assert isinstance(rate, float)

    @pytest.mark.slow
    def test_zero_rate_at_training_points(self):
        """Predicted rates at training points should be close to training rates."""
        gp = GaussianProcessCurve(reference_date=REF_DATE, kernel="rbf", length_scale=1.0, noise_variance=0.01)
        gp.fit(MATURITIES, RATES)

        for mat, actual in zip(MATURITIES, RATES):
            predicted = gp.zero_rate(mat)
            assert abs(predicted - actual) < 0.5, (
                f"Predicted {predicted} too far from actual {actual}"
            )

    @pytest.mark.slow
    def test_uncertainty_positive(self):
        gp = GaussianProcessCurve(reference_date=REF_DATE, kernel="rbf", length_scale=1.0, noise_variance=0.01)
        gp.fit(MATURITIES, RATES)

        for t in [0.5, 3.0, 15.0]:
            s = gp.uncertainty(t)
            assert s >= 0.0


# ---------------------------------------------------------------------------
# NeuralNetCurve
# ---------------------------------------------------------------------------


class TestNeuralNetCurve:
    """Tests for the neural-network yield-curve fitter."""

    @pytest.mark.slow
    def test_constructor(self):
        nn = NeuralNetCurve(reference_date=REF_DATE, hidden_layers=[16, 8], learning_rate=0.01, max_iter=100)
        assert nn is not None

    @pytest.mark.slow
    def test_fit_returns_result(self):
        nn = NeuralNetCurve(reference_date=REF_DATE, hidden_layers=[16, 8], learning_rate=0.01, max_iter=100)
        result = nn.fit(MATURITIES, RATES)

        assert isinstance(result, NNCurveResult)
        assert hasattr(result, "training_loss")
        assert isinstance(result.training_loss, float)

    @pytest.mark.slow
    def test_zero_rate_returns_float(self):
        nn = NeuralNetCurve(reference_date=REF_DATE, hidden_layers=[16, 8], learning_rate=0.01, max_iter=100)
        nn.fit(MATURITIES, RATES)

        for t in [0.5, 2.0, 7.0]:
            rate = nn.zero_rate(t)
            assert isinstance(rate, float)

    @pytest.mark.slow
    def test_training_loss_positive(self):
        nn = NeuralNetCurve(reference_date=REF_DATE, hidden_layers=[16, 8], learning_rate=0.01, max_iter=100)
        result = nn.fit(MATURITIES, RATES)

        assert result.training_loss >= 0.0


# ---------------------------------------------------------------------------
# CurveAutoencoder
# ---------------------------------------------------------------------------


class TestCurveAutoencoder:
    """Tests for the curve autoencoder (dimensionality reduction)."""

    @pytest.mark.slow
    def test_constructor(self):
        ae = CurveAutoencoder(input_dim=6, latent_dim=2, hidden_dim=8)
        assert ae is not None

    @pytest.mark.slow
    def test_fit_returns_result(self):
        curves = [
            [4.5, 4.3, 4.0, 3.8, 3.5, 3.2],
            [4.6, 4.4, 4.1, 3.9, 3.6, 3.3],
            [4.4, 4.2, 3.9, 3.7, 3.4, 3.1],
            [4.7, 4.5, 4.2, 4.0, 3.7, 3.4],
        ]
        ae = CurveAutoencoder(input_dim=6, latent_dim=2, hidden_dim=8)
        result = ae.fit(curves)

        assert isinstance(result, AutoencoderResult)

    @pytest.mark.slow
    def test_encode_dimension(self):
        curves = [
            [4.5, 4.3, 4.0, 3.8, 3.5, 3.2],
            [4.6, 4.4, 4.1, 3.9, 3.6, 3.3],
            [4.4, 4.2, 3.9, 3.7, 3.4, 3.1],
            [4.7, 4.5, 4.2, 4.0, 3.7, 3.4],
        ]
        ae = CurveAutoencoder(input_dim=6, latent_dim=2, hidden_dim=8)
        ae.fit(curves)

        encoded = ae.encode(curves)
        assert len(encoded) == len(curves)
        for vec in encoded:
            assert len(vec) == 2

    @pytest.mark.slow
    def test_decode_round_trip(self):
        curves = [
            [4.5, 4.3, 4.0, 3.8, 3.5, 3.2],
            [4.6, 4.4, 4.1, 3.9, 3.6, 3.3],
            [4.4, 4.2, 3.9, 3.7, 3.4, 3.1],
            [4.7, 4.5, 4.2, 4.0, 3.7, 3.4],
        ]
        ae = CurveAutoencoder(input_dim=6, latent_dim=2, hidden_dim=8)
        ae.fit(curves)

        encoded = ae.encode(curves)
        decoded = ae.decode(encoded)

        assert len(decoded) == len(curves)
        for original, reconstructed in zip(curves, decoded):
            assert len(reconstructed) == len(original)


# ---------------------------------------------------------------------------
# GPVolSurface
# ---------------------------------------------------------------------------


class TestGPVolSurface:
    """Tests for Gaussian Process volatility surface fitting."""

    @pytest.fixture
    def vol_data(self):
        maturities = [0.25, 0.25, 0.5, 0.5, 1.0, 1.0]
        strikes = [90.0, 110.0, 90.0, 110.0, 90.0, 110.0]
        vols = [0.25, 0.20, 0.23, 0.18, 0.22, 0.17]
        return maturities, strikes, vols

    @pytest.mark.slow
    def test_constructor(self):
        gp = GPVolSurface(kernel="rbf_2d")
        assert gp is not None

    @pytest.mark.slow
    def test_fit_returns_result(self, vol_data):
        maturities, strikes, vols = vol_data
        gp = GPVolSurface(kernel="rbf_2d")
        result = gp.fit(maturities, strikes, vols)

        assert isinstance(result, GPVolSurfaceResult)

    @pytest.mark.slow
    def test_volatility_returns_float(self, vol_data):
        maturities, strikes, vols = vol_data
        gp = GPVolSurface(kernel="rbf_2d")
        gp.fit(maturities, strikes, vols)

        vol = gp.volatility(strike=100.0, expiry=0.5)
        assert isinstance(vol, float)

    @pytest.mark.slow
    def test_volatility_positive(self, vol_data):
        maturities, strikes, vols = vol_data
        gp = GPVolSurface(kernel="rbf_2d")
        gp.fit(maturities, strikes, vols)

        for strike, expiry in [(100.0, 0.3), (100.0, 0.75)]:
            v = gp.volatility(strike=strike, expiry=expiry)
            assert v >= 0.0, "Predicted volatility should be non-negative"


# ---------------------------------------------------------------------------
# NeuralNetVolSurface
# ---------------------------------------------------------------------------


class TestNeuralNetVolSurface:
    """Tests for neural network volatility surface fitting."""

    @pytest.fixture
    def vol_data(self):
        maturities = [0.25, 0.25, 0.5, 0.5, 1.0, 1.0]
        strikes = [90.0, 110.0, 90.0, 110.0, 90.0, 110.0]
        vols = [0.25, 0.20, 0.23, 0.18, 0.22, 0.17]
        return maturities, strikes, vols

    @pytest.mark.slow
    def test_constructor(self):
        nn = NeuralNetVolSurface(hidden_layers=[16, 8], max_iter=50)
        assert nn is not None

    @pytest.mark.slow
    def test_fit_returns_result(self, vol_data):
        maturities, strikes, vols = vol_data
        nn = NeuralNetVolSurface(hidden_layers=[16, 8], max_iter=50)
        result = nn.fit(maturities, strikes, vols)

        assert isinstance(result, NNVolSurfaceResult)

    @pytest.mark.slow
    def test_volatility_returns_float(self, vol_data):
        maturities, strikes, vols = vol_data
        nn = NeuralNetVolSurface(hidden_layers=[16, 8], max_iter=50)
        nn.fit(maturities, strikes, vols)

        vol = nn.volatility(strike=100.0, expiry=0.5)
        assert isinstance(vol, float)

    @pytest.mark.slow
    def test_volatility_at_training_points(self, vol_data):
        maturities, strikes, vols = vol_data
        nn = NeuralNetVolSurface(hidden_layers=[16, 8], max_iter=50)
        nn.fit(maturities, strikes, vols)

        for s, e in zip(strikes, maturities):
            vol = nn.volatility(strike=s, expiry=e)
            assert isinstance(vol, float)
