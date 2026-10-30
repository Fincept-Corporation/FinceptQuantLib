"""Shared fixtures for ML tests."""

import math
import random
import pytest


@pytest.fixture
def sample_X():
    """Simple 6x2 feature matrix."""
    return [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0],
    ]


@pytest.fixture
def sample_y():
    """Simple regression targets for 6 samples."""
    return [2.5, 5.5, 8.5, 11.5, 14.5, 17.5]


@pytest.fixture
def binary_y():
    """Binary classification targets."""
    return [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


@pytest.fixture
def large_X():
    """Larger 50x3 feature matrix with known structure."""
    random.seed(42)
    X = []
    for i in range(50):
        x1 = i * 0.1
        x2 = math.sin(i * 0.2)
        x3 = random.gauss(0, 1)
        X.append([x1, x2, x3])
    return X


@pytest.fixture
def large_y():
    """Regression targets for large_X: y = 2*x1 + 0.5*x2 + noise."""
    random.seed(42)
    y = []
    for i in range(50):
        x1 = i * 0.1
        x2 = math.sin(i * 0.2)
        noise = random.gauss(0, 0.1)
        y.append(2.0 * x1 + 0.5 * x2 + noise)
    return y


@pytest.fixture
def time_series_data():
    """Simple time series with a regime shift at index 50."""
    random.seed(123)
    data = []
    for i in range(100):
        if i < 50:
            data.append(random.gauss(0.0, 1.0))
        else:
            data.append(random.gauss(3.0, 1.5))
    return data


@pytest.fixture
def price_series():
    """Simulated price series for technical indicators."""
    random.seed(99)
    prices = [100.0]
    for _ in range(99):
        prices.append(prices[-1] * (1 + random.gauss(0.001, 0.02)))
    return prices


@pytest.fixture
def volume_series():
    """Simulated volume series."""
    random.seed(77)
    return [max(1000, int(random.gauss(10000, 3000))) for _ in range(100)]
