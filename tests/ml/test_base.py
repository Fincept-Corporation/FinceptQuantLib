"""Tests for finceptquantlib.ml._base module."""

import pytest
from typing import Any, Dict, List

from finceptquantlib.ml._base import (
    ModelType,
    FitResult,
    PredictResult,
    Estimator,
    Classifier,
    Clusterer,
    Transformer,
)


# ---------------------------------------------------------------------------
# ModelType enum
# ---------------------------------------------------------------------------

class TestModelType:
    def test_classifier_value(self):
        assert ModelType.CLASSIFIER.value == "classifier"

    def test_regressor_value(self):
        assert ModelType.REGRESSOR.value == "regressor"

    def test_transformer_value(self):
        assert ModelType.TRANSFORMER.value == "transformer"

    def test_clusterer_value(self):
        assert ModelType.CLUSTERER.value == "clusterer"

    def test_all_members(self):
        names = {m.name for m in ModelType}
        assert names == {"CLASSIFIER", "REGRESSOR", "TRANSFORMER", "CLUSTERER"}


# ---------------------------------------------------------------------------
# FitResult dataclass
# ---------------------------------------------------------------------------

class TestFitResult:
    def test_construction(self):
        result = FitResult(converged=True, iterations=10, loss=0.01, message="ok")
        assert result.converged is True
        assert result.iterations == 10
        assert result.loss == pytest.approx(0.01, abs=1e-10)
        assert result.message == "ok"

    def test_default_message(self):
        result = FitResult(converged=False, iterations=5, loss=1.5)
        assert result.message == ""

    def test_not_converged(self):
        result = FitResult(converged=False, iterations=100, loss=99.9, message="fail")
        assert result.converged is False
        assert result.iterations == 100


# ---------------------------------------------------------------------------
# PredictResult dataclass
# ---------------------------------------------------------------------------

class TestPredictResult:
    def test_construction(self):
        result = PredictResult(predictions=[1.0, 2.0, 3.0], metadata={"key": "val"})
        assert result.predictions == [1.0, 2.0, 3.0]
        assert result.metadata == {"key": "val"}

    def test_default_metadata(self):
        result = PredictResult(predictions=[0.5])
        assert result.metadata == {}

    def test_empty_predictions(self):
        result = PredictResult(predictions=[])
        assert result.predictions == []


# ---------------------------------------------------------------------------
# Concrete implementations for testing ABCs
# ---------------------------------------------------------------------------

class DummyEstimator(Estimator):
    """Minimal concrete Estimator for testing."""

    def __init__(self):
        self._fitted = False
        self._coeff = 0.0

    def fit(self, X: List[List[float]], y: List[float]) -> FitResult:
        self._fitted = True
        self._coeff = sum(y) / len(y) if y else 0.0
        return FitResult(converged=True, iterations=1, loss=0.0, message="done")

    def predict(self, X: List[List[float]]) -> PredictResult:
        preds = [self._coeff for _ in X]
        return PredictResult(predictions=preds)

    def to_dict(self) -> Dict[str, Any]:
        return {"coeff": self._coeff, "fitted": self._fitted}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DummyEstimator":
        obj = cls()
        obj._coeff = data["coeff"]
        obj._fitted = data["fitted"]
        return obj


class DummyClassifier(Classifier):
    """Minimal concrete Classifier for testing."""

    def __init__(self):
        self._classes = [0.0, 1.0]

    def fit(self, X: List[List[float]], y: List[float]) -> FitResult:
        return FitResult(converged=True, iterations=1, loss=0.0)

    def predict(self, X: List[List[float]]) -> PredictResult:
        return PredictResult(predictions=[0.0] * len(X))

    def predict_proba(self, X: List[List[float]]) -> List[List[float]]:
        return [[0.5, 0.5] for _ in X]

    def to_dict(self) -> Dict[str, Any]:
        return {"classes": self._classes}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DummyClassifier":
        obj = cls()
        obj._classes = data["classes"]
        return obj


class DummyTransformer(Transformer):
    """Minimal concrete Transformer for testing."""

    def __init__(self):
        self._offset: List[float] = []
        self._fit_called = False

    def fit(self, X: List[List[float]]) -> None:
        self._fit_called = True
        if X:
            n = len(X)
            m = len(X[0])
            self._offset = [sum(X[i][j] for i in range(n)) / n for j in range(m)]
        else:
            self._offset = []

    def transform(self, X: List[List[float]]) -> List[List[float]]:
        return [[row[j] - self._offset[j] for j in range(len(row))] for row in X]

    def inverse_transform(self, X: List[List[float]]) -> List[List[float]]:
        return [[row[j] + self._offset[j] for j in range(len(row))] for row in X]


class DummyClusterer(Clusterer):
    """Minimal concrete Clusterer for testing."""

    def __init__(self):
        self._fitted = False

    def fit(self, X: List[List[float]]) -> Any:
        self._fitted = True
        return {"n_clusters": 2}

    def predict(self, X: List[List[float]]) -> List[int]:
        return [0] * len(X)


# ---------------------------------------------------------------------------
# Estimator ABC tests
# ---------------------------------------------------------------------------

class TestEstimator:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Estimator()

    def test_fit(self):
        est = DummyEstimator()
        X = [[1.0, 2.0], [3.0, 4.0]]
        y = [10.0, 20.0]
        result = est.fit(X, y)
        assert isinstance(result, FitResult)
        assert result.converged is True
        assert result.iterations == 1

    def test_predict(self):
        est = DummyEstimator()
        est.fit([[1.0], [2.0]], [5.0, 15.0])
        result = est.predict([[3.0], [4.0]])
        assert isinstance(result, PredictResult)
        assert len(result.predictions) == 2
        assert result.predictions[0] == pytest.approx(10.0, abs=1e-10)

    def test_to_dict(self):
        est = DummyEstimator()
        est.fit([[1.0]], [7.0])
        d = est.to_dict()
        assert isinstance(d, dict)
        assert d["coeff"] == pytest.approx(7.0, abs=1e-10)
        assert d["fitted"] is True

    def test_from_dict(self):
        data = {"coeff": 42.0, "fitted": True}
        est = DummyEstimator.from_dict(data)
        assert isinstance(est, DummyEstimator)
        assert est._coeff == pytest.approx(42.0, abs=1e-10)
        assert est._fitted is True

    def test_round_trip_serialization(self):
        est = DummyEstimator()
        est.fit([[1.0, 2.0]], [3.0])
        d = est.to_dict()
        est2 = DummyEstimator.from_dict(d)
        assert est2.to_dict() == d


# ---------------------------------------------------------------------------
# Classifier ABC tests
# ---------------------------------------------------------------------------

class TestClassifier:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Classifier()

    def test_extends_estimator(self):
        clf = DummyClassifier()
        assert isinstance(clf, Estimator)

    def test_predict_proba(self):
        clf = DummyClassifier()
        clf.fit([[1.0], [2.0]], [0.0, 1.0])
        probs = clf.predict_proba([[3.0], [4.0]])
        assert len(probs) == 2
        assert len(probs[0]) == 2
        assert probs[0][0] == pytest.approx(0.5, abs=1e-10)
        assert probs[0][1] == pytest.approx(0.5, abs=1e-10)

    def test_predict(self):
        clf = DummyClassifier()
        clf.fit([[1.0]], [1.0])
        result = clf.predict([[1.0]])
        assert isinstance(result, PredictResult)


# ---------------------------------------------------------------------------
# Transformer ABC tests
# ---------------------------------------------------------------------------

class TestTransformer:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Transformer()

    def test_fit(self):
        t = DummyTransformer()
        t.fit([[1.0, 2.0], [3.0, 4.0]])
        assert t._fit_called is True

    def test_transform(self):
        t = DummyTransformer()
        X = [[1.0, 2.0], [3.0, 4.0]]
        t.fit(X)
        result = t.transform(X)
        assert len(result) == 2
        # mean of col 0 is 2.0, mean of col 1 is 3.0
        assert result[0][0] == pytest.approx(-1.0, abs=1e-10)
        assert result[0][1] == pytest.approx(-1.0, abs=1e-10)
        assert result[1][0] == pytest.approx(1.0, abs=1e-10)
        assert result[1][1] == pytest.approx(1.0, abs=1e-10)

    def test_fit_transform_calls_fit_then_transform(self):
        t = DummyTransformer()
        X = [[1.0, 2.0], [3.0, 4.0]]
        result = t.fit_transform(X)
        assert t._fit_called is True
        # Same result as calling fit then transform separately
        t2 = DummyTransformer()
        t2.fit(X)
        expected = t2.transform(X)
        for i in range(len(result)):
            for j in range(len(result[0])):
                assert result[i][j] == pytest.approx(expected[i][j], abs=1e-10)

    def test_inverse_transform(self):
        t = DummyTransformer()
        X = [[10.0, 20.0], [30.0, 40.0]]
        transformed = t.fit_transform(X)
        recovered = t.inverse_transform(transformed)
        for i in range(len(X)):
            for j in range(len(X[0])):
                assert recovered[i][j] == pytest.approx(X[i][j], abs=1e-10)


# ---------------------------------------------------------------------------
# Clusterer ABC tests
# ---------------------------------------------------------------------------

class TestClusterer:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Clusterer()

    def test_fit(self):
        c = DummyClusterer()
        result = c.fit([[1.0, 2.0], [3.0, 4.0]])
        assert c._fitted is True
        assert result == {"n_clusters": 2}

    def test_predict(self):
        c = DummyClusterer()
        c.fit([[1.0], [2.0]])
        labels = c.predict([[1.0], [2.0], [3.0]])
        assert isinstance(labels, list)
        assert len(labels) == 3
        assert all(isinstance(lbl, int) for lbl in labels)
