"""
Power Transformations

Provides variance-stabilizing power transformations for financial data:

- ``BoxCoxTransformer``: Box-Cox transform for strictly positive data.
  Commonly applied to prices, volumes, or absolute returns.
- ``PowerTransformer``: Yeo-Johnson transform that handles negative
  values. Useful for returns and P&L data.

Both transformers learn optimal lambda parameters by maximizing the
profile log-likelihood, producing approximately Gaussian output
distributions suitable for downstream ML models.
"""

from __future__ import annotations

import math
from typing import List, Optional

from finceptquantlib.ml._base import Transformer


class BoxCoxTransformer(Transformer):
    """Box-Cox power transformation for strictly positive data.

    The Box-Cox transformation is defined as::

        y = (x^lambda - 1) / lambda    if lambda != 0
        y = log(x)                     if lambda == 0

    If ``lambda_param`` is None, the optimal lambda is found by
    maximizing the profile log-likelihood over the training data.

    Parameters
    ----------
    lambda_param : Optional[float]
        Fixed lambda parameter. If None, lambda is optimized during
        ``fit()``. Default is None.

    Attributes
    ----------
    lambda_ : Optional[float]
        The lambda parameter (either user-specified or optimized).

    Notes
    -----
    All input values must be strictly positive (x > 0). For data
    containing zeros, consider adding a small constant before
    transformation.

    References
    ----------
    .. [1] Box, G.E.P. and Cox, D.R. (1964). An analysis of
       transformations. *Journal of the Royal Statistical Society,
       Series B*, 26(2), 211-252.

    Examples
    --------
    >>> bc = BoxCoxTransformer()
    >>> X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
    >>> bc.fit(X)
    >>> transformed = bc.transform(X)
    >>> original = bc.inverse_transform(transformed)
    """

    def __init__(
        self, lambda_param: Optional[float] = None
    ) -> None:
        self._user_lambda = lambda_param
        self.lambda_: Optional[float] = lambda_param

    def fit(self, X: List[List[float]]) -> None:
        """Learn the optimal lambda from training data.

        If lambda_param was specified at construction, this stores it
        directly. Otherwise, optimizes lambda for each feature
        independently (stored as a single lambda for all features
        via the mean of per-feature optima).

        Parameters
        ----------
        X : List[List[float]]
            Training data (n_samples x n_features). All values must
            be strictly positive.

        Raises
        ------
        ValueError
            If X is empty or contains non-positive values.
        """
        if not X:
            raise ValueError("Cannot fit on empty data.")
        self._validate_positive(X)

        if self._user_lambda is not None:
            self.lambda_ = self._user_lambda
        else:
            # Optimize lambda by maximizing profile log-likelihood
            # Use a single lambda across all features (common practice)
            self.lambda_ = self._optimize_lambda(X)

    def transform(self, X: List[List[float]]) -> List[List[float]]:
        """Apply Box-Cox transformation.

        Parameters
        ----------
        X : List[List[float]]
            Data to transform. All values must be strictly positive.

        Returns
        -------
        List[List[float]]
            Box-Cox transformed data.

        Raises
        ------
        RuntimeError
            If transform is called before fit.
        ValueError
            If data contains non-positive values.
        """
        if self.lambda_ is None:
            raise RuntimeError("Must call fit() before transform().")
        self._validate_positive(X)
        lam = self.lambda_
        result: List[List[float]] = []
        for row in X:
            new_row: List[float] = []
            for val in row:
                new_row.append(self._box_cox(val, lam))
            result.append(new_row)
        return result

    def inverse_transform(
        self, X: List[List[float]]
    ) -> List[List[float]]:
        """Reverse Box-Cox transformation.

        Parameters
        ----------
        X : List[List[float]]
            Transformed data.

        Returns
        -------
        List[List[float]]
            Data in original scale.

        Raises
        ------
        RuntimeError
            If inverse_transform is called before fit.
        """
        if self.lambda_ is None:
            raise RuntimeError(
                "Must call fit() before inverse_transform()."
            )
        lam = self.lambda_
        result: List[List[float]] = []
        for row in X:
            new_row: List[float] = []
            for val in row:
                new_row.append(self._inv_box_cox(val, lam))
            result.append(new_row)
        return result

    @staticmethod
    def _box_cox(x: float, lam: float) -> float:
        """Apply Box-Cox transform to a single value.

        Parameters
        ----------
        x : float
            Value to transform (must be > 0).
        lam : float
            Lambda parameter.

        Returns
        -------
        float
            Transformed value.
        """
        if abs(lam) < 1e-10:
            return math.log(x)
        return (x ** lam - 1.0) / lam

    @staticmethod
    def _inv_box_cox(y: float, lam: float) -> float:
        """Inverse Box-Cox transform for a single value.

        Parameters
        ----------
        y : float
            Transformed value.
        lam : float
            Lambda parameter.

        Returns
        -------
        float
            Original-scale value.
        """
        if abs(lam) < 1e-10:
            return math.exp(y)
        inner = y * lam + 1.0
        if inner <= 0.0:
            # Clamp to a small positive value to avoid domain error
            inner = 1e-15
        return inner ** (1.0 / lam)

    @staticmethod
    def _validate_positive(X: List[List[float]]) -> None:
        """Validate that all values are strictly positive.

        Parameters
        ----------
        X : List[List[float]]
            Data to validate.

        Raises
        ------
        ValueError
            If any value is non-positive.
        """
        for i, row in enumerate(X):
            for j, val in enumerate(row):
                if val <= 0.0:
                    raise ValueError(
                        f"Box-Cox requires strictly positive values. "
                        f"Found {val} at row {i}, col {j}."
                    )

    def _optimize_lambda(
        self, X: List[List[float]]
    ) -> float:
        """Find optimal lambda by maximizing profile log-likelihood.

        Uses a grid search followed by golden section refinement.

        The profile log-likelihood for the Box-Cox transform is::

            L(lambda) = -(n/2) * log(var(y_lambda))
                        + (lambda - 1) * sum(log(x_i))

        Parameters
        ----------
        X : List[List[float]]
            Training data.

        Returns
        -------
        float
            Optimal lambda value.
        """
        n = len(X)
        if n == 0:
            return 1.0

        # Pool all values for a single lambda
        all_vals: List[float] = []
        for row in X:
            all_vals.extend(row)
        total = len(all_vals)
        if total == 0:
            return 1.0

        # Precompute sum of log(x) for the likelihood
        sum_log_x = sum(math.log(v) for v in all_vals)

        def neg_log_likelihood(lam: float) -> float:
            """Negative profile log-likelihood for lambda.

            Parameters
            ----------
            lam : float
                Lambda parameter to evaluate.

            Returns
            -------
            float
                Negative log-likelihood (to minimize).
            """
            transformed: List[float] = []
            for v in all_vals:
                transformed.append(
                    BoxCoxTransformer._box_cox(v, lam)
                )
            # Mean
            mean_t = sum(transformed) / total
            # Variance
            var_t = sum(
                (t - mean_t) ** 2 for t in transformed
            ) / total
            if var_t <= 0.0:
                var_t = 1e-15
            ll = (
                -0.5 * total * math.log(var_t)
                + (lam - 1.0) * sum_log_x
            )
            return -ll

        # Grid search over lambda in [-2, 3]
        best_lam = 1.0
        best_nll = neg_log_likelihood(1.0)
        for lam_int in range(-20, 31):
            lam = lam_int / 10.0
            nll = neg_log_likelihood(lam)
            if nll < best_nll:
                best_nll = nll
                best_lam = lam

        # Golden section refinement around best grid point
        a = best_lam - 0.15
        b = best_lam + 0.15
        gr = (math.sqrt(5.0) + 1.0) / 2.0
        tol = 1e-6
        for _ in range(50):
            if b - a < tol:
                break
            c = b - (b - a) / gr
            d = a + (b - a) / gr
            if neg_log_likelihood(c) < neg_log_likelihood(d):
                b = d
            else:
                a = c

        return (a + b) / 2.0


class PowerTransformer(Transformer):
    """Yeo-Johnson power transformation.

    Unlike Box-Cox, the Yeo-Johnson transformation handles both
    positive and negative values, making it suitable for financial
    returns and P&L data.

    The Yeo-Johnson transform is defined as::

        If x >= 0 and lambda != 0:
            y = ((x + 1)^lambda - 1) / lambda
        If x >= 0 and lambda == 0:
            y = log(x + 1)
        If x < 0 and lambda != 2:
            y = -((-x + 1)^(2 - lambda) - 1) / (2 - lambda)
        If x < 0 and lambda == 2:
            y = -log(-x + 1)

    Parameters
    ----------
    method : str
        Transformation method. Currently only ``"yeo-johnson"`` is
        supported. Default is ``"yeo-johnson"``.

    Attributes
    ----------
    lambda_ : Optional[float]
        Optimized lambda parameter learned during fit.

    References
    ----------
    .. [1] Yeo, I.K. and Johnson, R.A. (2000). A new family of power
       transformations to improve normality or symmetry.
       *Biometrika*, 87(4), 954-959.

    Examples
    --------
    >>> pt = PowerTransformer(method="yeo-johnson")
    >>> X = [[-2.0], [-1.0], [0.0], [1.0], [2.0]]
    >>> pt.fit(X)
    >>> transformed = pt.transform(X)
    """

    def __init__(self, method: str = "yeo-johnson") -> None:
        if method != "yeo-johnson":
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Only 'yeo-johnson' is supported."
            )
        self.method = method
        self.lambda_: Optional[float] = None

    def fit(self, X: List[List[float]]) -> None:
        """Learn the optimal lambda by maximizing log-likelihood.

        Parameters
        ----------
        X : List[List[float]]
            Training data (n_samples x n_features).

        Raises
        ------
        ValueError
            If X is empty.
        """
        if not X:
            raise ValueError("Cannot fit on empty data.")
        self.lambda_ = self._optimize_lambda(X)

    def transform(self, X: List[List[float]]) -> List[List[float]]:
        """Apply Yeo-Johnson transformation.

        Parameters
        ----------
        X : List[List[float]]
            Data to transform.

        Returns
        -------
        List[List[float]]
            Transformed data.

        Raises
        ------
        RuntimeError
            If transform is called before fit.
        """
        if self.lambda_ is None:
            raise RuntimeError("Must call fit() before transform().")
        lam = self.lambda_
        result: List[List[float]] = []
        for row in X:
            new_row: List[float] = []
            for val in row:
                new_row.append(self._yeo_johnson(val, lam))
            result.append(new_row)
        return result

    def inverse_transform(
        self, X: List[List[float]]
    ) -> List[List[float]]:
        """Reverse Yeo-Johnson transformation.

        Parameters
        ----------
        X : List[List[float]]
            Transformed data.

        Returns
        -------
        List[List[float]]
            Data in original scale.

        Raises
        ------
        RuntimeError
            If inverse_transform is called before fit.
        """
        if self.lambda_ is None:
            raise RuntimeError(
                "Must call fit() before inverse_transform()."
            )
        lam = self.lambda_
        result: List[List[float]] = []
        for row in X:
            new_row: List[float] = []
            for val in row:
                new_row.append(self._inv_yeo_johnson(val, lam))
            result.append(new_row)
        return result

    @staticmethod
    def _yeo_johnson(x: float, lam: float) -> float:
        """Apply Yeo-Johnson transform to a single value.

        Parameters
        ----------
        x : float
            Value to transform.
        lam : float
            Lambda parameter.

        Returns
        -------
        float
            Transformed value.
        """
        if x >= 0.0:
            if abs(lam) < 1e-10:
                return math.log(x + 1.0)
            return ((x + 1.0) ** lam - 1.0) / lam
        else:
            if abs(lam - 2.0) < 1e-10:
                return -math.log(-x + 1.0)
            return -((-x + 1.0) ** (2.0 - lam) - 1.0) / (2.0 - lam)

    @staticmethod
    def _inv_yeo_johnson(y: float, lam: float) -> float:
        """Inverse Yeo-Johnson transform for a single value.

        Parameters
        ----------
        y : float
            Transformed value.
        lam : float
            Lambda parameter.

        Returns
        -------
        float
            Original-scale value.
        """
        if y >= 0.0:
            # Came from x >= 0 branch
            if abs(lam) < 1e-10:
                return math.exp(y) - 1.0
            inner = y * lam + 1.0
            if inner <= 0.0:
                inner = 1e-15
            return inner ** (1.0 / lam) - 1.0
        else:
            # Came from x < 0 branch
            if abs(lam - 2.0) < 1e-10:
                return 1.0 - math.exp(-y)
            inner = -y * (2.0 - lam) + 1.0
            if inner <= 0.0:
                inner = 1e-15
            return 1.0 - inner ** (1.0 / (2.0 - lam))

    def _optimize_lambda(
        self, X: List[List[float]]
    ) -> float:
        """Find optimal lambda by maximizing profile log-likelihood.

        The Yeo-Johnson log-likelihood is::

            L(lambda) = -(n/2) * log(var(y_lambda))
                        + (lambda - 1) * sum(sign(x_i) * log(|x_i| + 1))

        Uses grid search with golden section refinement.

        Parameters
        ----------
        X : List[List[float]]
            Training data.

        Returns
        -------
        float
            Optimal lambda value.
        """
        # Pool all values
        all_vals: List[float] = []
        for row in X:
            all_vals.extend(row)
        total = len(all_vals)
        if total == 0:
            return 1.0

        # Precompute sum of sign(x)*log(|x|+1)
        sum_sign_log: float = 0.0
        for v in all_vals:
            if v >= 0.0:
                sum_sign_log += math.log(v + 1.0)
            else:
                sum_sign_log -= math.log(-v + 1.0)

        def neg_log_likelihood(lam: float) -> float:
            """Negative profile log-likelihood for lambda.

            Parameters
            ----------
            lam : float
                Lambda parameter to evaluate.

            Returns
            -------
            float
                Negative log-likelihood (to minimize).
            """
            transformed = [
                PowerTransformer._yeo_johnson(v, lam)
                for v in all_vals
            ]
            mean_t = sum(transformed) / total
            var_t = sum(
                (t - mean_t) ** 2 for t in transformed
            ) / total
            if var_t <= 0.0:
                var_t = 1e-15
            ll = (
                -0.5 * total * math.log(var_t)
                + (lam - 1.0) * sum_sign_log
            )
            return -ll

        # Grid search over lambda in [-2, 3]
        best_lam = 1.0
        best_nll = neg_log_likelihood(1.0)
        for lam_int in range(-20, 31):
            lam = lam_int / 10.0
            nll = neg_log_likelihood(lam)
            if nll < best_nll:
                best_nll = nll
                best_lam = lam

        # Golden section refinement
        a = best_lam - 0.15
        b = best_lam + 0.15
        gr = (math.sqrt(5.0) + 1.0) / 2.0
        tol = 1e-6
        for _ in range(50):
            if b - a < tol:
                break
            c = b - (b - a) / gr
            d = a + (b - a) / gr
            if neg_log_likelihood(c) < neg_log_likelihood(d):
                b = d
            else:
                a = c

        return (a + b) / 2.0
