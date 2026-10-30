"""
Tree-Based Regression Models

Decision tree and gradient boosted regression implemented in
pure Python. Trees use MSE impurity criterion with configurable
depth and leaf constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from finceptquantlib.ml._base import Estimator, FitResult, PredictResult
from finceptquantlib.ml._metrics import mean_squared_error


# ---------------------------------------------------------------------------
# Simple LCG pseudo-random number generator (deterministic)
# ---------------------------------------------------------------------------

class _LCG:
    """Linear congruential generator for deterministic randomness.

    Parameters
    ----------
    seed : int
        Initial seed value.
    """

    def __init__(self, seed: int = 42) -> None:
        self._state = seed & 0xFFFFFFFF

    def next_int(self) -> int:
        """Return next pseudo-random 32-bit integer."""
        self._state = (
            (1103515245 * self._state + 12345) & 0x7FFFFFFF
        )
        return self._state

    def next_float(self) -> float:
        """Return next pseudo-random float in [0, 1)."""
        return self.next_int() / 2147483648.0

    def sample_indices(
        self, n: int, k: int, replace: bool = False
    ) -> List[int]:
        """Sample k indices from range(n).

        Parameters
        ----------
        n : int
            Population size.
        k : int
            Sample size.
        replace : bool
            If True, sample with replacement.

        Returns
        -------
        List[int]
            Sampled indices.
        """
        if replace:
            return [self.next_int() % n for _ in range(k)]
        # Fisher-Yates partial shuffle
        pool = list(range(n))
        k = min(k, n)
        for i in range(k):
            j = i + self.next_int() % (n - i)
            pool[i], pool[j] = pool[j], pool[i]
        return pool[:k]


# ---------------------------------------------------------------------------
# Tree node dataclass
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    """A single node in a regression decision tree.

    Attributes
    ----------
    feature_index : Optional[int]
        Feature used for splitting (None for leaf nodes).
    threshold : Optional[float]
        Split threshold (None for leaf nodes).
    left : Optional[TreeNode]
        Left child (samples where feature <= threshold).
    right : Optional[TreeNode]
        Right child (samples where feature > threshold).
    value : float
        Predicted value (mean of targets at this node).
    n_samples : int
        Number of training samples reaching this node.
    impurity : float
        MSE impurity at this node.
    """

    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    value: float = 0.0
    n_samples: int = 0
    impurity: float = 0.0


@dataclass
class TreeResult(FitResult):
    """Result of fitting a decision tree regressor.

    Attributes
    ----------
    n_leaves : int
        Number of leaf nodes in the fitted tree.
    max_depth_reached : int
        Maximum depth reached during tree construction.
    feature_importances : List[float]
        Normalised feature importances based on total impurity
        reduction weighted by node sample count.
    """

    n_leaves: int = 0
    max_depth_reached: int = 0
    feature_importances: List[float] = field(
        default_factory=list
    )


# ---------------------------------------------------------------------------
# Decision Tree Regressor
# ---------------------------------------------------------------------------

class DecisionTreeRegressor(Estimator):
    """Regression decision tree using MSE splitting criterion.

    Recursively partitions the feature space by choosing splits
    that maximally reduce the weighted mean squared error.

    Parameters
    ----------
    max_depth : int
        Maximum tree depth. Default is 5.
    min_samples_split : int
        Minimum samples required to attempt a split. Default 10.
    min_samples_leaf : int
        Minimum samples required in each leaf. Default is 5.
    min_impurity_decrease : float
        Minimum impurity decrease required for a split. Default 0.0.
    max_features : Optional[int]
        If set, consider only this many randomly chosen features
        per split (used by random forest). None means all features.
    random_state : Optional[int]
        Random seed for feature subsampling.
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        min_impurity_decrease: float = 0.0,
        max_features: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self._root: Optional[TreeNode] = None
        self._n_features: int = 0
        self._feature_importances: List[float] = []
        self._fitted: bool = False
        self._rng: Optional[_LCG] = (
            _LCG(random_state)
            if random_state is not None
            else None
        )

    @staticmethod
    def _mse(values: List[float]) -> float:
        """Compute MSE of a set of values around their mean.

        Parameters
        ----------
        values : List[float]
            Target values.

        Returns
        -------
        float
            Variance (MSE w.r.t. mean).
        """
        n = len(values)
        if n == 0:
            return 0.0
        mean = sum(values) / n
        return sum((v - mean) ** 2 for v in values) / n

    def _find_best_split(
        self,
        X: List[List[float]],
        y: List[float],
        indices: List[int],
    ) -> Tuple[
        Optional[int], Optional[float], Optional[List[int]],
        Optional[List[int]], float
    ]:
        """Find the best binary split for the given sample indices.

        Parameters
        ----------
        X : List[List[float]]
            Full feature matrix.
        y : List[float]
            Full target vector.
        indices : List[int]
            Indices of samples at the current node.

        Returns
        -------
        Tuple
            (feature_index, threshold, left_indices,
             right_indices, impurity_decrease)
            or (None, None, None, None, 0.0) if no valid split.
        """
        n = len(indices)
        parent_mse = self._mse([y[i] for i in indices])
        best_decrease = self.min_impurity_decrease
        best_feature: Optional[int] = None
        best_threshold: Optional[float] = None
        best_left: Optional[List[int]] = None
        best_right: Optional[List[int]] = None

        # Determine which features to evaluate
        if (
            self.max_features is not None
            and self._rng is not None
            and self.max_features < self._n_features
        ):
            feature_indices = self._rng.sample_indices(
                self._n_features, self.max_features
            )
        else:
            feature_indices = list(range(self._n_features))

        for feat in feature_indices:
            # Sort indices by this feature
            sorted_idx = sorted(
                indices, key=lambda i: X[i][feat]
            )
            vals = [X[i][feat] for i in sorted_idx]
            targets = [y[i] for i in sorted_idx]

            # Incremental MSE computation
            left_sum = 0.0
            left_sq_sum = 0.0
            left_n = 0
            total_sum = sum(targets)
            total_sq = sum(t * t for t in targets)

            for k in range(n - 1):
                val = targets[k]
                left_sum += val
                left_sq_sum += val * val
                left_n += 1

                # Skip if same feature value as next
                if vals[k] == vals[k + 1]:
                    continue

                right_n = n - left_n
                if (
                    left_n < self.min_samples_leaf
                    or right_n < self.min_samples_leaf
                ):
                    continue

                # MSE = E[X^2] - (E[X])^2
                left_mse = (
                    left_sq_sum / left_n
                    - (left_sum / left_n) ** 2
                )
                right_sum = total_sum - left_sum
                right_sq = total_sq - left_sq_sum
                right_mse = (
                    right_sq / right_n
                    - (right_sum / right_n) ** 2
                )

                weighted = (
                    left_n * left_mse + right_n * right_mse
                ) / n
                decrease = parent_mse - weighted

                if decrease > best_decrease:
                    best_decrease = decrease
                    best_feature = feat
                    best_threshold = (
                        vals[k] + vals[k + 1]
                    ) / 2.0
                    best_left = sorted_idx[: left_n]
                    best_right = sorted_idx[left_n:]

        return (
            best_feature,
            best_threshold,
            best_left,
            best_right,
            best_decrease,
        )

    def _build_tree(
        self,
        X: List[List[float]],
        y: List[float],
        indices: List[int],
        depth: int,
    ) -> Tuple[TreeNode, int, int]:
        """Recursively build the decision tree.

        Parameters
        ----------
        X : List[List[float]]
            Full feature matrix.
        y : List[float]
            Full target vector.
        indices : List[int]
            Sample indices at this node.
        depth : int
            Current depth.

        Returns
        -------
        Tuple[TreeNode, int, int]
            (node, n_leaves, max_depth_reached)
        """
        n = len(indices)
        targets = [y[i] for i in indices]
        node_value = sum(targets) / n if n > 0 else 0.0
        node_impurity = self._mse(targets)

        node = TreeNode(
            value=node_value,
            n_samples=n,
            impurity=node_impurity,
        )

        # Check stopping criteria
        if (
            depth >= self.max_depth
            or n < self.min_samples_split
            or node_impurity < 1e-14
        ):
            return node, 1, depth

        (
            feat_idx,
            threshold,
            left_idx,
            right_idx,
            decrease,
        ) = self._find_best_split(X, y, indices)

        if feat_idx is None:
            return node, 1, depth

        # Record feature importance (weighted impurity decrease)
        self._feature_importances[feat_idx] += (
            decrease * n
        )

        node.feature_index = feat_idx
        node.threshold = threshold

        left_node, left_leaves, left_depth = self._build_tree(
            X, y, left_idx, depth + 1  # type: ignore[arg-type]
        )
        right_node, right_leaves, right_depth = self._build_tree(
            X, y, right_idx, depth + 1  # type: ignore[arg-type]
        )
        node.left = left_node
        node.right = right_node

        return (
            node,
            left_leaves + right_leaves,
            max(left_depth, right_depth),
        )

    def fit(
        self, X: List[List[float]], y: List[float]
    ) -> TreeResult:
        """Fit decision tree regressor.

        Parameters
        ----------
        X : List[List[float]]
            Feature matrix (n_samples x n_features).
        y : List[float]
            Target values (n_samples,).

        Returns
        -------
        TreeResult
            Tree fitting diagnostics including feature importances.
        """
        n = len(X)
        if n == 0:
            raise ValueError("X must not be empty")
        self._n_features = len(X[0])
        if len(y) != n:
            raise ValueError(
                "X and y must have the same number of samples"
            )
        self._feature_importances = [0.0] * self._n_features

        self._root, n_leaves, max_depth = self._build_tree(
            X, y, list(range(n)), 0
        )
        self._fitted = True

        # Normalise feature importances
        total_imp = sum(self._feature_importances)
        if total_imp > 0:
            self._feature_importances = [
                fi / total_imp
                for fi in self._feature_importances
            ]

        y_pred = self.predict(X).predictions
        mse = mean_squared_error(y, y_pred)

        return TreeResult(
            converged=True,
            iterations=1,
            loss=mse,
            message="Decision tree fit completed",
            n_leaves=n_leaves,
            max_depth_reached=max_depth,
            feature_importances=self._feature_importances[:],
        )

    @staticmethod
    def _predict_single(
        node: TreeNode, row: List[float]
    ) -> float:
        """Traverse the tree to predict a single sample.

        Parameters
        ----------
        node : TreeNode
            Root of the tree or subtree.
        row : List[float]
            Feature vector.

        Returns
        -------
        float
            Predicted value.
        """
        while node.feature_index is not None:
            if row[node.feature_index] <= node.threshold:  # type: ignore[operator]
                node = node.left  # type: ignore[assignment]
            else:
                node = node.right  # type: ignore[assignment]
        return node.value

    def predict(
        self, X: List[List[float]]
    ) -> PredictResult:
        """Predict target values for new samples.

        Parameters
        ----------
        X : List[List[float]]
            Feature matrix (n_samples x n_features).

        Returns
        -------
        PredictResult
            Predicted values.
        """
        if not self._fitted or self._root is None:
            raise RuntimeError("Model has not been fitted")
        preds = [
            self._predict_single(self._root, row) for row in X
        ]
        return PredictResult(predictions=preds)

    def _tree_to_dict(
        self, node: Optional[TreeNode]
    ) -> Optional[Dict[str, Any]]:
        """Recursively serialize a tree node.

        Parameters
        ----------
        node : Optional[TreeNode]
            Tree node to serialize.

        Returns
        -------
        Optional[Dict[str, Any]]
            Serialized node or None.
        """
        if node is None:
            return None
        return {
            "feature_index": node.feature_index,
            "threshold": node.threshold,
            "value": node.value,
            "n_samples": node.n_samples,
            "impurity": node.impurity,
            "left": self._tree_to_dict(node.left),
            "right": self._tree_to_dict(node.right),
        }

    @staticmethod
    def _tree_from_dict(
        data: Optional[Dict[str, Any]]
    ) -> Optional[TreeNode]:
        """Recursively reconstruct a tree node from dict.

        Parameters
        ----------
        data : Optional[Dict[str, Any]]
            Serialized node data.

        Returns
        -------
        Optional[TreeNode]
            Reconstructed node or None.
        """
        if data is None:
            return None
        node = TreeNode(
            feature_index=data.get("feature_index"),
            threshold=data.get("threshold"),
            value=data.get("value", 0.0),
            n_samples=data.get("n_samples", 0),
            impurity=data.get("impurity", 0.0),
        )
        node.left = DecisionTreeRegressor._tree_from_dict(
            data.get("left")
        )
        node.right = DecisionTreeRegressor._tree_from_dict(
            data.get("right")
        )
        return node

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to dictionary.

        Returns
        -------
        Dict[str, Any]
            JSON-serializable model parameters.
        """
        return {
            "type": "DecisionTreeRegressor",
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_impurity_decrease": self.min_impurity_decrease,
            "n_features": self._n_features,
            "feature_importances": self._feature_importances,
            "tree": self._tree_to_dict(self._root),
            "fitted": self._fitted,
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any]
    ) -> "DecisionTreeRegressor":
        """Reconstruct model from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Serialized model parameters.

        Returns
        -------
        DecisionTreeRegressor
            Reconstructed model.
        """
        model = cls(
            max_depth=data.get("max_depth", 5),
            min_samples_split=data.get("min_samples_split", 10),
            min_samples_leaf=data.get("min_samples_leaf", 5),
            min_impurity_decrease=data.get(
                "min_impurity_decrease", 0.0
            ),
        )
        model._n_features = data.get("n_features", 0)
        model._feature_importances = data.get(
            "feature_importances", []
        )
        model._root = cls._tree_from_dict(data.get("tree"))
        model._fitted = data.get("fitted", False)
        return model


# ---------------------------------------------------------------------------
# Gradient Boosted Regressor
# ---------------------------------------------------------------------------

class GradientBoostedRegressor(Estimator):
    """Gradient Boosted Regression Trees.

    Sequentially fits shallow decision trees to the negative
    gradient (pseudo-residuals) of the squared error loss.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds. Default is 100.
    learning_rate : float
        Shrinkage factor applied to each tree. Default is 0.1.
    max_depth : int
        Maximum depth per tree. Default is 3.
    subsample : float
        Fraction of samples used per tree (stochastic gradient
        boosting). Default is 1.0 (no subsampling).
    min_samples_leaf : int
        Minimum samples per leaf in each tree. Default is 5.
    random_state : int
        Random seed for reproducibility. Default is 42.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        subsample: float = 1.0,
        min_samples_leaf: int = 5,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self._f0: float = 0.0
        self._trees: List[DecisionTreeRegressor] = []
        self._feature_importances: List[float] = []
        self._loss_history: List[float] = []
        self._fitted: bool = False

    def fit(
        self, X: List[List[float]], y: List[float]
    ) -> FitResult:
        """Fit gradient boosted regression model.

        Parameters
        ----------
        X : List[List[float]]
            Feature matrix (n_samples x n_features).
        y : List[float]
            Target values (n_samples,).

        Returns
        -------
        FitResult
            Fitting result with training loss history in metadata.
        """
        n = len(X)
        if n == 0:
            raise ValueError("X must not be empty")
        n_features = len(X[0])
        if len(y) != n:
            raise ValueError(
                "X and y must have the same number of samples"
            )

        rng = _LCG(self.random_state)

        # Initialize F_0 = mean(y)
        self._f0 = sum(y) / n
        predictions = [self._f0] * n
        self._trees = []
        self._loss_history = []
        imp_accum = [0.0] * n_features

        for m in range(self.n_estimators):
            # Pseudo-residuals: r_im = y_i - F_{m-1}(x_i)
            residuals = [
                y[i] - predictions[i] for i in range(n)
            ]

            # Subsample if requested
            if self.subsample < 1.0:
                k = max(1, int(self.subsample * n))
                sample_idx = rng.sample_indices(n, k)
                X_sub = [X[i] for i in sample_idx]
                r_sub = [residuals[i] for i in sample_idx]
            else:
                X_sub = X
                r_sub = residuals

            # Fit a shallow tree to the residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=2,
                min_samples_leaf=self.min_samples_leaf,
                random_state=rng.next_int(),
            )
            tree_result = tree.fit(X_sub, r_sub)
            self._trees.append(tree)

            # Accumulate feature importances
            for j in range(
                min(n_features, len(tree_result.feature_importances))
            ):
                imp_accum[j] += tree_result.feature_importances[j]

            # Update predictions: F_m = F_{m-1} + lr * tree_m
            tree_preds = tree.predict(X).predictions
            for i in range(n):
                predictions[i] += (
                    self.learning_rate * tree_preds[i]
                )

            mse = sum(
                (y[i] - predictions[i]) ** 2 for i in range(n)
            ) / n
            self._loss_history.append(mse)

        # Normalise feature importances
        total_imp = sum(imp_accum)
        if total_imp > 0:
            self._feature_importances = [
                v / total_imp for v in imp_accum
            ]
        else:
            self._feature_importances = [0.0] * n_features

        self._fitted = True
        final_loss = (
            self._loss_history[-1]
            if self._loss_history
            else 0.0
        )
        return FitResult(
            converged=True,
            iterations=self.n_estimators,
            loss=final_loss,
            message=(
                f"GBR fit completed with "
                f"{self.n_estimators} estimators"
            ),
        )

    @property
    def feature_importances(self) -> List[float]:
        """Average feature importances across all trees.

        Returns
        -------
        List[float]
            Normalised feature importances.
        """
        return self._feature_importances

    @property
    def loss_history(self) -> List[float]:
        """Training loss after each boosting round.

        Returns
        -------
        List[float]
            MSE at each iteration.
        """
        return self._loss_history

    def predict(
        self, X: List[List[float]]
    ) -> PredictResult:
        """Predict target values for new samples.

        Parameters
        ----------
        X : List[List[float]]
            Feature matrix (n_samples x n_features).

        Returns
        -------
        PredictResult
            Predicted values.
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted")
        n = len(X)
        preds = [self._f0] * n
        for tree in self._trees:
            tree_preds = tree.predict(X).predictions
            for i in range(n):
                preds[i] += (
                    self.learning_rate * tree_preds[i]
                )
        return PredictResult(
            predictions=preds,
            metadata={
                "loss_history": self._loss_history,
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
            "type": "GradientBoostedRegressor",
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
            "f0": self._f0,
            "trees": [t.to_dict() for t in self._trees],
            "feature_importances": self._feature_importances,
            "loss_history": self._loss_history,
            "fitted": self._fitted,
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any]
    ) -> "GradientBoostedRegressor":
        """Reconstruct model from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Serialized model parameters.

        Returns
        -------
        GradientBoostedRegressor
            Reconstructed model.
        """
        model = cls(
            n_estimators=data.get("n_estimators", 100),
            learning_rate=data.get("learning_rate", 0.1),
            max_depth=data.get("max_depth", 3),
            subsample=data.get("subsample", 1.0),
            min_samples_leaf=data.get("min_samples_leaf", 5),
            random_state=data.get("random_state", 42),
        )
        model._f0 = data.get("f0", 0.0)
        model._trees = [
            DecisionTreeRegressor.from_dict(td)
            for td in data.get("trees", [])
        ]
        model._feature_importances = data.get(
            "feature_importances", []
        )
        model._loss_history = data.get("loss_history", [])
        model._fitted = data.get("fitted", False)
        return model
