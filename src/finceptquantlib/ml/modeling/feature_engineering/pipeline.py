"""
Feature Pipeline

Composable feature transformation pipeline that chains multiple
``Transformer`` steps. Each step is a named ``(name, transformer)``
pair executed in sequence.

Classes
-------
- FeaturePipeline: Sequential transformer composition
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from finceptquantlib.ml._base import Transformer


class FeaturePipeline(Transformer):
    """Sequential pipeline of named Transformer steps.

    Each step is a ``(name, transformer)`` tuple. During ``fit``,
    each transformer is fit on the output of the previous step.
    During ``transform``, each transformer transforms the data
    sequentially.

    Parameters
    ----------
    steps : List[Tuple[str, Transformer]]
        Named transformer steps to execute in order.

    Attributes
    ----------
    steps : List[Tuple[str, Transformer]]
        The pipeline steps.

    Examples
    --------
    >>> from finceptquantlib.ml._base import Transformer
    >>> class ScaleBy2(Transformer):
    ...     def fit(self, X):
    ...         pass
    ...     def transform(self, X):
    ...         return [[v * 2 for v in row] for row in X]
    ...     def inverse_transform(self, X):
    ...         return [[v / 2 for v in row] for row in X]
    >>> pipe = FeaturePipeline([("scale", ScaleBy2())])
    >>> pipe.fit_transform([[1.0, 2.0], [3.0, 4.0]])
    [[2.0, 4.0], [6.0, 8.0]]
    """

    def __init__(self, steps: List[Tuple[str, Transformer]]) -> None:
        self.steps = steps
        self._is_fitted: bool = False

    def fit(self, X: List[List[float]]) -> None:
        """Fit each transformer step sequentially.

        Each step is fit on the output of the previous step's
        transform.

        Parameters
        ----------
        X : List[List[float]]
            Training data (n_samples x n_features).
        """
        data = X
        for name, transformer in self.steps:
            transformer.fit(data)
            data = transformer.transform(data)
        self._is_fitted = True

    def transform(self, X: List[List[float]]) -> List[List[float]]:
        """Apply all transformer steps sequentially.

        Parameters
        ----------
        X : List[List[float]]
            Data to transform (n_samples x n_features).

        Returns
        -------
        List[List[float]]
            Transformed data after all steps.

        Raises
        ------
        RuntimeError
            If the pipeline has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "FeaturePipeline must be fitted before transform. "
                "Call fit() or fit_transform() first."
            )
        data = X
        for name, transformer in self.steps:
            data = transformer.transform(data)
        return data

    def fit_transform(
        self, X: List[List[float]]
    ) -> List[List[float]]:
        """Fit and transform in one pass.

        Parameters
        ----------
        X : List[List[float]]
            Data to fit and transform.

        Returns
        -------
        List[List[float]]
            Transformed data after all steps.
        """
        data = X
        for name, transformer in self.steps:
            data = transformer.fit_transform(data)
        self._is_fitted = True
        return data

    def inverse_transform(
        self, X: List[List[float]]
    ) -> List[List[float]]:
        """Apply inverse transformations in reverse order.

        Parameters
        ----------
        X : List[List[float]]
            Transformed data to invert.

        Returns
        -------
        List[List[float]]
            Data with all transformations reversed.

        Raises
        ------
        RuntimeError
            If the pipeline has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "FeaturePipeline must be fitted before "
                "inverse_transform."
            )
        data = X
        for name, transformer in reversed(self.steps):
            data = transformer.inverse_transform(data)
        return data

    def get_feature_names(self) -> List[str]:
        """Return the names of all pipeline steps.

        Returns
        -------
        List[str]
            Step names in pipeline order.
        """
        return [name for name, _ in self.steps]

    def get_step(self, name: str) -> Optional[Transformer]:
        """Retrieve a transformer step by name.

        Parameters
        ----------
        name : str
            Name of the step to retrieve.

        Returns
        -------
        Optional[Transformer]
            The transformer if found, ``None`` otherwise.
        """
        for step_name, transformer in self.steps:
            if step_name == name:
                return transformer
        return None

    def __repr__(self) -> str:
        step_strs = [
            f"  ('{name}', {transformer.__class__.__name__})"
            for name, transformer in self.steps
        ]
        joined = ",\n".join(step_strs)
        return f"FeaturePipeline([\n{joined}\n])"

    def __len__(self) -> int:
        return len(self.steps)
