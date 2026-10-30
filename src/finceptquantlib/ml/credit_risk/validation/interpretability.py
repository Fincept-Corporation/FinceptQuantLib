"""
Model Interpretability Tools

Provides model-agnostic and tree-specific interpretability methods:
- KernelSHAP: Shapley-value-based feature attribution
- TreeSHAP: Exact SHAP values for tree-based models
- Permutation Importance: Feature importance via score degradation
- Partial Dependence: Marginal effect of individual features

All operations use pure Python lists without external dependencies.
These methods are essential for regulatory model documentation and
explainability requirements (SR 11-7, BCBS 239).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, List, Optional


@dataclass
class SHAPResult:
    """Result of SHAP value computation.

    Attributes
    ----------
    shap_values : List[List[float]]
        SHAP values matrix (n_samples x n_features). Each value
        represents the marginal contribution of that feature to
        the prediction for that sample.
    base_value : float
        Expected model output over the background dataset.
    feature_names : Optional[List[str]]
        Feature names, if provided.
    """

    shap_values: List[List[float]]
    base_value: float
    feature_names: Optional[List[str]] = None


@dataclass
class PermutationImportanceResult:
    """Result of permutation importance analysis.

    Attributes
    ----------
    importances_mean : List[float]
        Mean importance for each feature (sorted descending).
    importances_std : List[float]
        Standard deviation of importance for each feature.
    feature_names : Optional[List[str]]
        Feature names in importance order.
    baseline_score : float
        Model score on unperturbed data.
    """

    importances_mean: List[float]
    importances_std: List[float]
    feature_names: Optional[List[str]] = None
    baseline_score: float = 0.0


@dataclass
class PDPResult:
    """Result of partial dependence computation.

    Attributes
    ----------
    feature_values : List[float]
        Grid of feature values evaluated.
    pdp_values : List[float]
        Average prediction at each grid point.
    ice_curves : Optional[List[List[float]]]
        Individual Conditional Expectation curves. Each inner list
        contains predictions for one sample across the grid.
    """

    feature_values: List[float]
    pdp_values: List[float]
    ice_curves: Optional[List[List[float]]] = None


def _combination(n: int, k: int) -> int:
    """Compute binomial coefficient C(n, k).

    Parameters
    ----------
    n : int
        Total number of items.
    k : int
        Number of items to choose.

    Returns
    -------
    int
        Binomial coefficient.
    """
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def _shap_kernel_weight(M: int, coalition_size: int) -> float:
    """Compute SHAP kernel weight for a coalition.

    The SHAP kernel weight is:
        w(|z|) = (M - 1) / (C(M, |z|) * |z| * (M - |z|))

    Parameters
    ----------
    M : int
        Total number of features.
    coalition_size : int
        Number of features in the coalition (|z|).

    Returns
    -------
    float
        Kernel weight. Returns large value for edge coalitions.
    """
    s = coalition_size
    if s == 0 or s == M:
        return 1e6  # Large weight for empty/full coalitions
    comb = _combination(M, s)
    if comb == 0:
        return 1e6
    return (M - 1.0) / (comb * s * (M - s))


def _solve_weighted_least_squares(
    Z: List[List[float]],
    y: List[float],
    weights: List[float],
) -> List[float]:
    """Solve weighted least squares: min_beta sum w_i (y_i - Z_i beta)^2.

    Uses the normal equations: (Z^T W Z) beta = Z^T W y
    with Gaussian elimination for the solve step.

    Parameters
    ----------
    Z : List[List[float]]
        Design matrix (n_coalitions x n_features).
    y : List[float]
        Target values (n_coalitions,).
    weights : List[float]
        Non-negative weights (n_coalitions,).

    Returns
    -------
    List[float]
        Coefficient vector (n_features,).
    """
    n = len(Z)
    if n == 0:
        return []
    m = len(Z[0])

    # Compute Z^T W Z (m x m)
    ZtWZ: List[List[float]] = [[0.0] * m for _ in range(m)]
    # Compute Z^T W y (m,)
    ZtWy: List[float] = [0.0] * m

    for i in range(n):
        w = weights[i]
        for j in range(m):
            for k in range(j, m):
                val = w * Z[i][j] * Z[i][k]
                ZtWZ[j][k] += val
                if j != k:
                    ZtWZ[k][j] += val
            ZtWy[j] += w * Z[i][j] * y[i]

    # Add small ridge for numerical stability
    for j in range(m):
        ZtWZ[j][j] += 1e-10

    # Gaussian elimination with partial pivoting
    aug: List[List[float]] = [
        ZtWZ[i][:] + [ZtWy[i]] for i in range(m)
    ]

    for col in range(m):
        # Partial pivoting
        max_row = col
        max_val = abs(aug[col][col])
        for row in range(col + 1, m):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row
        if max_row != col:
            aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        if abs(pivot) < 1e-15:
            continue

        for row in range(col + 1, m):
            factor = aug[row][col] / pivot
            for k in range(col, m + 1):
                aug[row][k] -= factor * aug[col][k]

    # Back substitution
    beta = [0.0] * m
    for i in range(m - 1, -1, -1):
        if abs(aug[i][i]) < 1e-15:
            beta[i] = 0.0
            continue
        s = aug[i][m]
        for j in range(i + 1, m):
            s -= aug[i][j] * beta[j]
        beta[i] = s / aug[i][i]

    return beta


def kernel_shap(
    model: Any,
    X: List[List[float]],
    background: List[List[float]],
    n_samples: int = 100,
    feature_names: Optional[List[str]] = None,
) -> SHAPResult:
    """Compute approximate SHAP values using KernelSHAP.

    KernelSHAP approximates Shapley values by sampling coalitions
    of features and solving a weighted least-squares problem. For
    excluded features, background distribution means are used.

    Parameters
    ----------
    model : Any
        Model with a ``predict(X)`` method returning an object
        with a ``predictions`` attribute.
    X : List[List[float]]
        Samples to explain (n_samples x n_features).
    background : List[List[float]]
        Background dataset for computing expectations
        (n_background x n_features).
    n_samples : int, optional
        Number of coalition samples per instance. Default is 100.
    feature_names : Optional[List[str]], optional
        Feature names for the result.

    Returns
    -------
    SHAPResult
        SHAP values for each sample and the base value.

    Raises
    ------
    ValueError
        If X or background is empty.
    """
    if len(X) == 0:
        raise ValueError("X must not be empty")
    if len(background) == 0:
        raise ValueError("background must not be empty")

    n_features = len(X[0])
    n_bg = len(background)

    # Compute background feature means
    bg_means: List[float] = [0.0] * n_features
    for row in background:
        for j in range(n_features):
            bg_means[j] += row[j]
    for j in range(n_features):
        bg_means[j] /= n_bg

    # Compute base value (mean prediction on background)
    bg_preds = model.predict(background).predictions
    base_value = sum(bg_preds) / len(bg_preds)

    rng = random.Random(42)

    all_shap: List[List[float]] = []

    for sample in X:
        if n_features == 1:
            # Single feature: SHAP = prediction - base
            pred = model.predict([sample]).predictions[0]
            all_shap.append([pred - base_value])
            continue

        # Generate coalition samples
        coalitions: List[List[int]] = []
        Z_design: List[List[float]] = []
        y_marginal: List[float] = []
        kernel_weights: List[float] = []

        # Always include empty and full coalitions
        coalitions.append([0] * n_features)
        coalitions.append([1] * n_features)

        # Sample random coalitions
        for _ in range(n_samples):
            coalition = [
                rng.randint(0, 1) for _ in range(n_features)
            ]
            coalitions.append(coalition)

        for coalition in coalitions:
            s = sum(coalition)

            # Create masked sample
            masked = [
                sample[j] if coalition[j] == 1 else bg_means[j]
                for j in range(n_features)
            ]

            # Predict
            pred = model.predict([masked]).predictions[0]

            Z_design.append([float(c) for c in coalition])
            y_marginal.append(pred - base_value)
            kernel_weights.append(
                _shap_kernel_weight(n_features, s)
            )

        # Solve WLS for SHAP values
        shap_vals = _solve_weighted_least_squares(
            Z_design, y_marginal, kernel_weights
        )
        all_shap.append(shap_vals)

    return SHAPResult(
        shap_values=all_shap,
        base_value=base_value,
        feature_names=feature_names,
    )


def _tree_shap_recurse(
    node: Any,
    sample: List[float],
    feature_set: List[bool],
    proportion: float,
) -> List[float]:
    """Recursively compute TreeSHAP values for one sample.

    Traverses the tree, distributing marginal contributions
    of each split feature using interventional TreeSHAP.

    Parameters
    ----------
    node : Any
        Current tree node. Must have attributes:
        - ``is_leaf`` (bool): Whether this is a leaf node.
        - ``value`` (float): Prediction value (for leaves).
        - ``feature_index`` (int): Split feature (for internal).
        - ``threshold`` (float): Split threshold (for internal).
        - ``left`` (node): Left child (for internal).
        - ``right`` (node): Right child (for internal).
        - ``n_samples`` (int): Number of training samples.
    sample : List[float]
        Input sample to explain.
    feature_set : List[bool]
        Which features are "known" (True = known).
    proportion : float
        Current sample proportion reaching this node.

    Returns
    -------
    List[float]
        SHAP contribution for each feature at this node.
    """
    n_features = len(sample)
    contributions = [0.0] * n_features

    if node.is_leaf:
        return contributions

    feat_idx = node.feature_index
    left = node.left
    right = node.right

    left_n = left.n_samples if hasattr(left, "n_samples") else 1
    right_n = right.n_samples if hasattr(right, "n_samples") else 1
    total_n = left_n + right_n
    left_prop = left_n / total_n if total_n > 0 else 0.5
    right_prop = right_n / total_n if total_n > 0 else 0.5

    if feature_set[feat_idx]:
        # Feature is known: follow the decision path
        if sample[feat_idx] <= node.threshold:
            sub = _tree_shap_recurse(
                left, sample, feature_set, proportion
            )
        else:
            sub = _tree_shap_recurse(
                right, sample, feature_set, proportion
            )
        for j in range(n_features):
            contributions[j] += sub[j]
    else:
        # Feature is unknown: explore both branches
        # weighted by training sample proportions
        feature_set_with = feature_set[:]
        feature_set_with[feat_idx] = True

        # Conditional path
        if sample[feat_idx] <= node.threshold:
            cond_val = _get_subtree_value(left, sample, feature_set)
        else:
            cond_val = _get_subtree_value(
                right, sample, feature_set
            )

        # Marginal path
        marg_val = (
            left_prop
            * _get_subtree_value(left, sample, feature_set)
            + right_prop
            * _get_subtree_value(right, sample, feature_set)
        )

        contributions[feat_idx] += (cond_val - marg_val) * proportion

        # Recurse with feature now known
        left_sub = _tree_shap_recurse(
            left, sample, feature_set_with,
            proportion * left_prop,
        )
        right_sub = _tree_shap_recurse(
            right, sample, feature_set_with,
            proportion * right_prop,
        )
        for j in range(n_features):
            contributions[j] += left_sub[j] + right_sub[j]

    return contributions


def _get_subtree_value(
    node: Any,
    sample: List[float],
    feature_set: List[bool],
) -> float:
    """Get expected prediction from a subtree.

    For known features, follow the decision path.
    For unknown features, take weighted average of children.

    Parameters
    ----------
    node : Any
        Tree node.
    sample : List[float]
        Input sample.
    feature_set : List[bool]
        Which features are known.

    Returns
    -------
    float
        Expected prediction value.
    """
    if node.is_leaf:
        return node.value

    feat_idx = node.feature_index
    left = node.left
    right = node.right

    if feature_set[feat_idx]:
        if sample[feat_idx] <= node.threshold:
            return _get_subtree_value(left, sample, feature_set)
        else:
            return _get_subtree_value(right, sample, feature_set)
    else:
        left_n = (
            left.n_samples if hasattr(left, "n_samples") else 1
        )
        right_n = (
            right.n_samples if hasattr(right, "n_samples") else 1
        )
        total_n = left_n + right_n
        left_prop = left_n / total_n if total_n > 0 else 0.5
        right_prop = right_n / total_n if total_n > 0 else 0.5
        return (
            left_prop
            * _get_subtree_value(left, sample, feature_set)
            + right_prop
            * _get_subtree_value(right, sample, feature_set)
        )


def _tree_shap_single_tree(
    root: Any,
    sample: List[float],
    n_features: int,
) -> List[float]:
    """Compute TreeSHAP values for one sample on one tree.

    Parameters
    ----------
    root : Any
        Root node of the tree.
    sample : List[float]
        Input sample.
    n_features : int
        Number of features.

    Returns
    -------
    List[float]
        SHAP values for each feature.
    """
    feature_set = [False] * n_features
    return _tree_shap_recurse(root, sample, feature_set, 1.0)


def tree_shap(
    model: Any,
    X: List[List[float]],
    feature_names: Optional[List[str]] = None,
) -> SHAPResult:
    """Compute exact SHAP values for tree-based models.

    Uses the interventional TreeSHAP algorithm for models with
    a tree structure. Supports single decision trees (model._root),
    random forests, and gradient-boosted ensembles (model._trees).

    Parameters
    ----------
    model : Any
        Tree-based model with either:
        - ``_trees`` attribute: list of tree root nodes (ensemble).
        - ``_root`` attribute: single tree root node.
        Each node must have: ``is_leaf``, ``value``,
        ``feature_index``, ``threshold``, ``left``, ``right``,
        ``n_samples``.
    X : List[List[float]]
        Samples to explain (n_samples x n_features).
    feature_names : Optional[List[str]], optional
        Feature names for the result.

    Returns
    -------
    SHAPResult
        Exact SHAP values and base value.

    Raises
    ------
    ValueError
        If the model has no tree structure or X is empty.
    """
    if len(X) == 0:
        raise ValueError("X must not be empty")

    n_features = len(X[0])

    # Determine tree roots
    if hasattr(model, "_trees"):
        trees = model._trees
    elif hasattr(model, "_root"):
        trees = [model._root]
    else:
        raise ValueError(
            "Model must have a '_trees' or '_root' attribute "
            "containing tree node(s)"
        )

    if len(trees) == 0:
        raise ValueError("Model has no trees")

    n_trees = len(trees)

    # Compute base value: average prediction across all trees
    # at the root (marginal expectation with no features known)
    base_values: List[float] = []
    for root in trees:
        feature_set = [False] * n_features
        # Use a dummy sample (values don't matter when no features known)
        dummy = [0.0] * n_features
        base_values.append(
            _get_subtree_value(root, dummy, feature_set)
        )
    base_value = sum(base_values) / n_trees

    all_shap: List[List[float]] = []
    for sample in X:
        sample_shap = [0.0] * n_features
        for root in trees:
            tree_vals = _tree_shap_single_tree(
                root, sample, n_features
            )
            for j in range(n_features):
                sample_shap[j] += tree_vals[j] / n_trees
        all_shap.append(sample_shap)

    return SHAPResult(
        shap_values=all_shap,
        base_value=base_value,
        feature_names=feature_names,
    )


def permutation_importance(
    model: Any,
    X: List[List[float]],
    y: List[float],
    metric: Callable[[List[float], List[float]], float],
    n_repeats: int = 10,
    feature_names: Optional[List[str]] = None,
    random_state: int = 42,
) -> PermutationImportanceResult:
    """Compute permutation feature importance.

    Measures the decrease in model performance when each feature
    is randomly shuffled, breaking the relationship between the
    feature and the target. Higher importance means greater
    performance degradation.

    Parameters
    ----------
    model : Any
        Fitted model with a ``predict(X)`` method returning
        an object with a ``predictions`` attribute.
    X : List[List[float]]
        Feature matrix (n_samples x n_features).
    y : List[float]
        True target values (n_samples,).
    metric : Callable[[List[float], List[float]], float]
        Scoring function(y_true, y_pred) -> float. Higher is better.
    n_repeats : int, optional
        Number of random shuffles per feature. Default is 10.
    feature_names : Optional[List[str]], optional
        Feature names for labeling.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    PermutationImportanceResult
        Feature importances sorted by mean importance (descending).

    Raises
    ------
    ValueError
        If X is empty or X and y have mismatched lengths.
    """
    if len(X) == 0:
        raise ValueError("X must not be empty")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    n_samples = len(X)
    n_features = len(X[0])

    # Baseline score
    baseline_preds = model.predict(X).predictions
    baseline_score = metric(y, baseline_preds)

    rng = random.Random(random_state)

    # Per-feature importance scores
    raw_means: List[float] = []
    raw_stds: List[float] = []

    for j in range(n_features):
        scores: List[float] = []
        for _ in range(n_repeats):
            # Create shuffled copy of column j
            X_shuffled = [row[:] for row in X]
            col_values = [X[i][j] for i in range(n_samples)]
            rng.shuffle(col_values)
            for i in range(n_samples):
                X_shuffled[i][j] = col_values[i]

            # Score with shuffled feature
            shuffled_preds = model.predict(X_shuffled).predictions
            shuffled_score = metric(y, shuffled_preds)
            importance = baseline_score - shuffled_score
            scores.append(importance)

        mean_imp = sum(scores) / len(scores)
        if len(scores) > 1:
            var_imp = sum(
                (s - mean_imp) ** 2 for s in scores
            ) / (len(scores) - 1)
            std_imp = math.sqrt(var_imp)
        else:
            std_imp = 0.0

        raw_means.append(mean_imp)
        raw_stds.append(std_imp)

    # Sort by mean importance descending
    if feature_names is not None:
        names = list(feature_names)
    else:
        names = [f"feature_{j}" for j in range(n_features)]

    indices = sorted(
        range(n_features), key=lambda j: -raw_means[j]
    )

    sorted_means = [raw_means[j] for j in indices]
    sorted_stds = [raw_stds[j] for j in indices]
    sorted_names = [names[j] for j in indices]

    return PermutationImportanceResult(
        importances_mean=sorted_means,
        importances_std=sorted_stds,
        feature_names=sorted_names,
        baseline_score=baseline_score,
    )


def partial_dependence(
    model: Any,
    X: List[List[float]],
    feature_index: int,
    grid_resolution: int = 50,
    feature_name: Optional[str] = None,
) -> PDPResult:
    """Compute Partial Dependence Plot (PDP) values.

    PDP shows the marginal effect of a single feature on the
    model's prediction, averaged over all other features. ICE
    (Individual Conditional Expectation) curves show the effect
    for each individual sample.

    Parameters
    ----------
    model : Any
        Fitted model with a ``predict(X)`` method returning
        an object with a ``predictions`` attribute.
    X : List[List[float]]
        Feature matrix (n_samples x n_features).
    feature_index : int
        Index of the feature to compute dependence for.
    grid_resolution : int, optional
        Number of evenly spaced grid points. Default is 50.
    feature_name : Optional[str], optional
        Name of the feature for labeling.

    Returns
    -------
    PDPResult
        PDP values, grid points, and ICE curves.

    Raises
    ------
    ValueError
        If X is empty, feature_index is out of range, or
        grid_resolution < 2.
    """
    if len(X) == 0:
        raise ValueError("X must not be empty")
    n_samples = len(X)
    n_features = len(X[0])
    if feature_index < 0 or feature_index >= n_features:
        raise ValueError(
            f"feature_index {feature_index} out of range "
            f"[0, {n_features})"
        )
    if grid_resolution < 2:
        raise ValueError("grid_resolution must be at least 2")

    # Compute grid values
    col = [X[i][feature_index] for i in range(n_samples)]
    feat_min = min(col)
    feat_max = max(col)

    if feat_min == feat_max:
        grid_values = [feat_min]
    else:
        step = (feat_max - feat_min) / (grid_resolution - 1)
        grid_values = [
            feat_min + i * step for i in range(grid_resolution)
        ]

    # Compute PDP and ICE
    pdp_values: List[float] = []
    ice_curves: List[List[float]] = [
        [] for _ in range(n_samples)
    ]

    for grid_val in grid_values:
        # Replace feature_index with grid value for all samples
        X_modified = [row[:] for row in X]
        for i in range(n_samples):
            X_modified[i][feature_index] = grid_val

        preds = model.predict(X_modified).predictions

        # PDP = average prediction
        pdp_values.append(sum(preds) / n_samples)

        # ICE = individual predictions
        for i in range(n_samples):
            ice_curves[i].append(preds[i])

    return PDPResult(
        feature_values=grid_values,
        pdp_values=pdp_values,
        ice_curves=ice_curves,
    )
