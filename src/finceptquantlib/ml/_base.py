"""
Base Classes for Machine Learning Module

Provides abstract base classes that all ML models implement:
- Estimator: Base for supervised learning (fit/predict)
- Classifier: Extension with probability prediction
- Transformer: Base for feature transformers (fit/transform)
- Result dataclasses for structured outputs
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class ModelType(Enum):
    """Type of ML model."""

    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"
    TRANSFORMER = "transformer"
    CLUSTERER = "clusterer"


@dataclass
class FitResult:
    """Base result for model fitting.

    Attributes
    ----------
    converged : bool
        Whether the optimization converged.
    iterations : int
        Number of iterations performed.
    loss : float
        Final loss/objective value.
    message : str
        Status message from the optimizer.
    """

    converged: bool
    iterations: int
    loss: float
    message: str = ""


@dataclass
class PredictResult:
    """Base result for model predictions.

    Attributes
    ----------
    predictions : List[float]
        Predicted values.
    metadata : Dict[str, Any]
        Additional prediction metadata.
    """

    predictions: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class Estimator(ABC):
    """Abstract base class for all supervised ML estimators.

    All estimators implement fit/predict and serialization via
    to_dict/from_dict for JSON-friendly serialization.

    Methods
    -------
    fit(X, y)
        Train the model on features X and targets y.
    predict(X)
        Generate predictions for features X.
    to_dict()
        Serialize model parameters to a dictionary.
    from_dict(data)
        Reconstruct model from serialized dictionary.
    """

    @abstractmethod
    def fit(
        self, X: List[List[float]], y: List[float]
    ) -> FitResult:
        """Train the model.

        Parameters
        ----------
        X : List[List[float]]
            Feature matrix (n_samples x n_features).
        y : List[float]
            Target values (n_samples,).

        Returns
        -------
        FitResult
            Training result with convergence info.
        """
        ...

    @abstractmethod
    def predict(self, X: List[List[float]]) -> PredictResult:
        """Generate predictions.

        Parameters
        ----------
        X : List[List[float]]
            Feature matrix (n_samples x n_features).

        Returns
        -------
        PredictResult
            Predictions and metadata.
        """
        ...

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to dictionary.

        Returns
        -------
        Dict[str, Any]
            JSON-serializable model parameters.
        """
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Estimator":
        """Reconstruct model from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Serialized model parameters.

        Returns
        -------
        Estimator
            Reconstructed model instance.
        """
        ...


class Classifier(Estimator):
    """Abstract base class for classifiers.

    Extends Estimator with probability prediction.
    """

    @abstractmethod
    def predict_proba(
        self, X: List[List[float]]
    ) -> List[List[float]]:
        """Predict class probabilities.

        Parameters
        ----------
        X : List[List[float]]
            Feature matrix (n_samples x n_features).

        Returns
        -------
        List[List[float]]
            Probability matrix (n_samples x n_classes).
        """
        ...


class Transformer(ABC):
    """Abstract base class for feature transformers.

    Transformers learn parameters from data (fit) and apply
    transformations (transform). Supports inverse_transform
    for reversible transformations.
    """

    @abstractmethod
    def fit(self, X: List[List[float]]) -> None:
        """Learn transformation parameters from data.

        Parameters
        ----------
        X : List[List[float]]
            Training data (n_samples x n_features).
        """
        ...

    @abstractmethod
    def transform(
        self, X: List[List[float]]
    ) -> List[List[float]]:
        """Apply the learned transformation.

        Parameters
        ----------
        X : List[List[float]]
            Data to transform (n_samples x n_features).

        Returns
        -------
        List[List[float]]
            Transformed data.
        """
        ...

    def fit_transform(
        self, X: List[List[float]]
    ) -> List[List[float]]:
        """Fit and transform in one step.

        Parameters
        ----------
        X : List[List[float]]
            Data to fit and transform.

        Returns
        -------
        List[List[float]]
            Transformed data.
        """
        self.fit(X)
        return self.transform(X)

    @abstractmethod
    def inverse_transform(
        self, X: List[List[float]]
    ) -> List[List[float]]:
        """Reverse the transformation.

        Parameters
        ----------
        X : List[List[float]]
            Transformed data.

        Returns
        -------
        List[List[float]]
            Original-scale data.
        """
        ...


class Clusterer(ABC):
    """Abstract base class for clustering algorithms.

    Clusterers assign data points to groups without target labels.
    """

    @abstractmethod
    def fit(self, X: List[List[float]]) -> Any:
        """Fit the clustering model.

        Parameters
        ----------
        X : List[List[float]]
            Data matrix (n_samples x n_features).

        Returns
        -------
        Any
            Clustering result dataclass.
        """
        ...

    @abstractmethod
    def predict(self, X: List[List[float]]) -> List[int]:
        """Assign cluster labels to new data.

        Parameters
        ----------
        X : List[List[float]]
            Data matrix (n_samples x n_features).

        Returns
        -------
        List[int]
            Cluster labels for each sample.
        """
        ...
