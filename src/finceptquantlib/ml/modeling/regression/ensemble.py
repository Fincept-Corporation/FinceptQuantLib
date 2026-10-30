"""
Ensemble Regression Models

Random Forest regressor implemented in pure Python using
bootstrap aggregation of decision trees with random feature
subsets.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from finceptquantlib.ml._base import Estimator, FitResult, PredictResult
from finceptquantlib.ml._metrics import mean_squared_error, r_squared
from finceptquantlib.ml.modeling.regression.trees import (
    DecisionTreeRegressor,
    _LCG,
)


class RandomForestRegressor(Estimator):
    """Random Forest regression via bagging of decision trees.

    Each tree is trained on a bootstrap sample of the data using
    a random subset of features at each split. Predictions are
    the average across all trees.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest. Default is 100.
    max_depth : int
        Maximum depth of each tree. Default is 5.
    max_features : Optional[int]
        Number of features to consider at each split. If None,
        uses floor(sqrt(n_features)). Default is None.
    min_samples_leaf : int
        Minimum samples in each leaf node. Default is 5.
    random_state : int
        Seed for reproducibility. Default is 42.

    Examples
    --------
    >>> model = RandomForestRegressor(n_estimators=10, max_depth=3)
    >>> result = model.fit(
    ...     [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    ...     [1.0, 2.0, 3.0],
    ... )
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        max_features: Optional[int] = None,
        min_samples_leaf: int = 5,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self._trees: List[DecisionTreeRegressor] = []
        self._feature_importances: List[float] = []
        self._oob_score: Optional[float] = None
        self._fitted: bool = False

    def fit(
        self, X: List[List[float]], y: List[float]
    ) -> FitResult:
        """Fit random forest regression model.

        Each tree is trained on a bootstrap sample (sampling with
        replacement) using a random subset of features at every
        split. Out-of-bag (OOB) predictions are computed for each
        sample and used to estimate generalisation error.

        Parameters
        ----------
        X : List[List[float]]
            Feature matrix (n_samples x n_features).
        y : List[float]
            Target values (n_samples,).

        Returns
        -------
        FitResult
            Fitting result. The loss field contains the OOB MSE
            when available, otherwise the in-bag training MSE.

        Raises
        ------
        ValueError
            If X and y are incompatible.
        """
        n = len(X)
        if n == 0:
            raise ValueError("X must not be empty")
        n_features = len(X[0])
        if len(y) != n:
            raise ValueError(
                "X and y must have the same number of samples"
            )

        # Determine max_features
        mf = self.max_features
        if mf is None:
            mf = max(1, int(math.sqrt(n_features)))
        mf = min(mf, n_features)

        rng = _LCG(self.random_state)
        self._trees = []
        imp_accum = [0.0] * n_features

        # OOB tracking: for each sample accumulate predictions
        oob_sum = [0.0] * n
        oob_count = [0] * n

        for _ in range(self.n_estimators):
            # Bootstrap sample (with replacement)
            boot_idx = rng.sample_indices(n, n, replace=True)
            X_boot = [X[i] for i in boot_idx]
            y_boot = [y[i] for i in boot_idx]

            # Track which samples are out-of-bag
            in_bag = set(boot_idx)

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=2,
                min_samples_leaf=self.min_samples_leaf,
                max_features=mf,
                random_state=rng.next_int(),
            )
            tree_result = tree.fit(X_boot, y_boot)
            self._trees.append(tree)

            # Accumulate feature importances
            for j in range(
                min(
                    n_features,
                    len(tree_result.feature_importances),
                )
            ):
                imp_accum[j] += (
                    tree_result.feature_importances[j]
                )

            # OOB predictions
            oob_indices = [
                i for i in range(n) if i not in in_bag
            ]
            if oob_indices:
                oob_X = [X[i] for i in oob_indices]
                oob_preds = tree.predict(oob_X).predictions
                for idx, oi in enumerate(oob_indices):
                    oob_sum[oi] += oob_preds[idx]
                    oob_count[oi] += 1

        # Normalise feature importances
        total_imp = sum(imp_accum)
        if total_imp > 0:
            self._feature_importances = [
                v / total_imp for v in imp_accum
            ]
        else:
            self._feature_importances = [0.0] * n_features

        # Compute OOB score
        oob_true: List[float] = []
        oob_pred: List[float] = []
        for i in range(n):
            if oob_count[i] > 0:
                oob_true.append(y[i])
                oob_pred.append(oob_sum[i] / oob_count[i])

        if len(oob_true) > 1:
            oob_mse = mean_squared_error(oob_true, oob_pred)
            self._oob_score = r_squared(oob_true, oob_pred)
        else:
            oob_mse = None
            self._oob_score = None

        self._fitted = True

        # Compute training MSE as fallback
        if oob_mse is not None:
            loss = oob_mse
            msg = (
                f"Random forest fit completed with "
                f"{self.n_estimators} trees "
                f"(OOB R^2={self._oob_score:.4f})"
            )
        else:
            train_preds = self.predict(X).predictions
            loss = mean_squared_error(y, train_preds)
            msg = (
                f"Random forest fit completed with "
                f"{self.n_estimators} trees (no OOB score)"
            )

        return FitResult(
            converged=True,
            iterations=self.n_estimators,
            loss=loss,
            message=msg,
        )

    @property
    def feature_importances(self) -> List[float]:
        """Average feature importances across all trees.

        Returns
        -------
        List[float]
            Normalised feature importances summing to 1.
        """
        return self._feature_importances

    @property
    def oob_score(self) -> Optional[float]:
        """Out-of-bag R-squared score.

        Returns
        -------
        Optional[float]
            OOB R^2, or None if unavailable.
        """
        return self._oob_score

    def predict(
        self, X: List[List[float]]
    ) -> PredictResult:
        """Predict by averaging predictions of all trees.

        Parameters
        ----------
        X : List[List[float]]
            Feature matrix (n_samples x n_features).

        Returns
        -------
        PredictResult
            Averaged predictions.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted")
        n = len(X)
        n_trees = len(self._trees)
        sums = [0.0] * n

        for tree in self._trees:
            tree_preds = tree.predict(X).predictions
            for i in range(n):
                sums[i] += tree_preds[i]

        preds = [s / n_trees for s in sums]
        return PredictResult(
            predictions=preds,
            metadata={
                "n_trees": n_trees,
                "oob_score": self._oob_score,
                "feature_importances": (
                    self._feature_importances
                ),
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to dictionary.

        Returns
        -------
        Dict[str, Any]
            JSON-serializable model parameters.
        """
        return {
            "type": "RandomForestRegressor",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "max_features": self.max_features,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
            "trees": [t.to_dict() for t in self._trees],
            "feature_importances": self._feature_importances,
            "oob_score": self._oob_score,
            "fitted": self._fitted,
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any]
    ) -> "RandomForestRegressor":
        """Reconstruct model from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Serialized model parameters.

        Returns
        -------
        RandomForestRegressor
            Reconstructed model.
        """
        model = cls(
            n_estimators=data.get("n_estimators", 100),
            max_depth=data.get("max_depth", 5),
            max_features=data.get("max_features"),
            min_samples_leaf=data.get("min_samples_leaf", 5),
            random_state=data.get("random_state", 42),
        )
        model._trees = [
            DecisionTreeRegressor.from_dict(td)
            for td in data.get("trees", [])
        ]
        model._feature_importances = data.get(
            "feature_importances", []
        )
        model._oob_score = data.get("oob_score")
        model._fitted = data.get("fitted", False)
        return model
