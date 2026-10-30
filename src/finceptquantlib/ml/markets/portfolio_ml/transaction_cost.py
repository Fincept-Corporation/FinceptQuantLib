"""
Transaction Cost Modeling

Provides models for estimating and optimizing execution costs
in portfolio rebalancing:

- **TransactionCostModel**: Predicts total transaction costs
  decomposed into spread and market impact components, with
  Almgren-Chriss optimal execution scheduling.

All computations are pure Python with no external numerical
dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional

from finceptquantlib.ml._base import FitResult
from finceptquantlib.ml._matrix_utils import (
    mat_mul,
    mat_vec_mul,
    transpose,
)


# -------------------------------------------------------------------
# Result dataclass
# -------------------------------------------------------------------


@dataclass
class TransactionCostResult:
    """Result of transaction cost prediction.

    Attributes
    ----------
    predicted_costs : List[float]
        Total predicted cost per trade (in currency or bps).
    spread_component : List[float]
        Bid-ask spread component of cost per trade.
    impact_component : List[float]
        Market impact component of cost per trade.
    total_cost : float
        Aggregate total cost across all trades.
    """

    predicted_costs: List[float]
    spread_component: List[float]
    impact_component: List[float]
    total_cost: float


# -------------------------------------------------------------------
# Linear algebra helpers
# -------------------------------------------------------------------


def _ols_fit(
    X: List[List[float]], y: List[float]
) -> List[float]:
    """Ordinary least squares regression.

    Solves beta = (X'X)^{-1} X'y via Cholesky decomposition.

    Parameters
    ----------
    X : List[List[float]]
        Design matrix (n x k).
    y : List[float]
        Response vector (n,).

    Returns
    -------
    List[float]
        Coefficient vector (k,).
    """
    Xt = transpose(X)
    XtX = mat_mul(Xt, X)
    Xty = mat_vec_mul(Xt, y)

    k = len(XtX)
    # Regularize for numerical stability
    for i in range(k):
        XtX[i][i] += 1e-10

    # Cholesky decomposition
    L = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(i + 1):
            s = sum(L[i][m] * L[j][m] for m in range(j))
            if i == j:
                val = XtX[i][i] - s
                L[i][j] = math.sqrt(max(val, 1e-14))
            else:
                if L[j][j] > 0:
                    L[i][j] = (XtX[i][j] - s) / L[j][j]

    # Forward substitution: L z = Xty
    z = [0.0] * k
    for i in range(k):
        s = sum(L[i][m] * z[m] for m in range(i))
        if L[i][i] > 0:
            z[i] = (Xty[i] - s) / L[i][i]

    # Back substitution: L' beta = z
    Lt = transpose(L)
    beta = [0.0] * k
    for i in range(k - 1, -1, -1):
        s = sum(Lt[i][m] * beta[m] for m in range(i + 1, k))
        if Lt[i][i] > 0:
            beta[i] = (z[i] - s) / Lt[i][i]

    return beta


def _ols_predict(
    X: List[List[float]], beta: List[float]
) -> List[float]:
    """Predict using OLS coefficients.

    Parameters
    ----------
    X : List[List[float]]
        Feature matrix (n x k).
    beta : List[float]
        Coefficient vector (k,).

    Returns
    -------
    List[float]
        Predicted values (n,).
    """
    return mat_vec_mul(X, beta)


def _add_intercept(X: List[List[float]]) -> List[List[float]]:
    """Prepend a column of ones for intercept term."""
    return [[1.0] + row for row in X]


# -------------------------------------------------------------------
# TransactionCostModel
# -------------------------------------------------------------------


class TransactionCostModel:
    """Transaction cost prediction and optimal execution.

    Combines a linear prediction model for total trading costs
    with the Almgren-Chriss (2001) optimal execution framework
    for computing trade schedules that balance market impact
    against timing risk.

    Parameters
    ----------
    model : Any or None
        Custom estimator implementing fit/predict. If None,
        uses a built-in linear regression model.

    Notes
    -----
    Cost decomposition:
    - **Spread component**: half the bid-ask spread (immediate
      cost of crossing the spread)
    - **Impact component**: square-root impact model
      impact = sigma * sqrt(V / ADV) * sign(V)
      following Almgren et al. (2005)

    Optimal execution follows:
    Almgren, R. & Chriss, N. (2001). "Optimal execution of
    portfolio transactions." Journal of Risk 3(2):5-39.

    Examples
    --------
    >>> from finceptquantlib.ml.markets.portfolio_ml.transaction_cost import (
    ...     TransactionCostModel,
    ... )
    >>> features = [
    ...     [1000, 1e9, 1e6, 0.02, 0.001],
    ...     [500, 5e8, 5e5, 0.03, 0.002],
    ... ]
    >>> costs = [0.005, 0.008]
    >>> tcm = TransactionCostModel()
    >>> fit_result = tcm.fit(features, costs)
    >>> result = tcm.predict(features)
    """

    def __init__(self, model: Optional[Any] = None) -> None:
        self._custom_model = model
        self._beta: Optional[List[float]] = None
        self._fitted: bool = False

    def fit(
        self,
        features: List[List[float]],
        realized_costs: List[float],
    ) -> FitResult:
        """Fit the transaction cost prediction model.

        Parameters
        ----------
        features : List[List[float]]
            Feature matrix (n_trades x n_features). Typical
            features: trade_size, market_cap, avg_volume,
            volatility, bid_ask_spread.
        realized_costs : List[float]
            Realized total transaction costs per trade (n_trades,).

        Returns
        -------
        FitResult
            Fitting result with convergence info.
        """
        n = len(features)
        if n == 0:
            raise ValueError("No training data provided")

        if self._custom_model is not None:
            result = self._custom_model.fit(
                features, realized_costs
            )
            self._fitted = True
            return result

        # Built-in linear regression with intercept
        X_aug = _add_intercept(features)
        self._beta = _ols_fit(X_aug, realized_costs)

        # Compute training loss (MSE)
        predictions = _ols_predict(X_aug, self._beta)
        mse = sum(
            (p - y) ** 2
            for p, y in zip(predictions, realized_costs)
        ) / n

        self._fitted = True

        return FitResult(
            converged=True,
            iterations=1,
            loss=mse,
            message="Linear regression fit completed",
        )

    def predict(
        self, features: List[List[float]]
    ) -> TransactionCostResult:
        """Predict transaction costs with decomposition.

        The features are expected to include (in order or as a
        superset): trade_size, market_cap, avg_volume, volatility,
        bid_ask_spread. The decomposition uses:

        - Spread component: 0.5 * bid_ask_spread * |trade_size|
          (half-spread cost)
        - Impact component: volatility * sqrt(|trade_size| / adv)
          * sign(trade_size) (square-root market impact model)

        Parameters
        ----------
        features : List[List[float]]
            Feature matrix (n_trades x n_features). Expected
            column order: [trade_size, market_cap, avg_volume,
            volatility, bid_ask_spread, ...].

        Returns
        -------
        TransactionCostResult
            Predicted costs with spread/impact decomposition.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if not self._fitted:
            raise RuntimeError(
                "Model must be fitted before calling predict"
            )

        n = len(features)
        n_feat = len(features[0]) if features else 0

        # Total cost prediction
        if self._custom_model is not None:
            pred_result = self._custom_model.predict(features)
            predicted = (
                pred_result.predictions
                if hasattr(pred_result, "predictions")
                else pred_result
            )
        else:
            X_aug = _add_intercept(features)
            predicted = _ols_predict(X_aug, self._beta)

        # Cost decomposition
        # Expected feature columns:
        # [0] trade_size, [1] market_cap, [2] avg_volume,
        # [3] volatility, [4] bid_ask_spread
        spread_comp: List[float] = []
        impact_comp: List[float] = []

        for i in range(n):
            row = features[i]

            # Extract relevant features with safe defaults
            trade_size = row[0] if n_feat > 0 else 0.0
            adv = row[2] if n_feat > 2 else 1.0
            volatility = row[3] if n_feat > 3 else 0.0
            bid_ask = row[4] if n_feat > 4 else 0.0

            # Spread: half the bid-ask spread times trade size
            spread = 0.5 * abs(bid_ask) * abs(trade_size)
            spread_comp.append(spread)

            # Square-root impact model:
            # impact = sigma * sqrt(|V| / ADV) * sign(V)
            if adv > 0 and abs(trade_size) > 0:
                sign_v = 1.0 if trade_size >= 0 else -1.0
                impact = (
                    volatility
                    * math.sqrt(abs(trade_size) / adv)
                    * sign_v
                )
            else:
                impact = 0.0
            impact_comp.append(impact)

        total_cost = sum(abs(p) for p in predicted)

        return TransactionCostResult(
            predicted_costs=predicted,
            spread_component=spread_comp,
            impact_component=impact_comp,
            total_cost=total_cost,
        )

    def optimal_execution(
        self,
        total_shares: float,
        n_periods: int,
        volatility: float,
        adv: float,
        risk_aversion: float = 1.0,
    ) -> List[float]:
        """Compute Almgren-Chriss optimal execution schedule.

        Determines the optimal trading trajectory that balances
        market impact costs against timing (volatility) risk.

        The model assumes:
        - Temporary impact: g(v) = eta * v (linear in trade rate)
        - Permanent impact: h(v) = gamma * v
        - Timing risk proportional to sigma^2 * remaining inventory

        The optimal number of shares remaining at period j is:

            n_j = X * sinh(kappa * (N - j)) / sinh(kappa * N)

        where:
        - X = total_shares
        - N = n_periods
        - kappa = arccosh(0.5 * (delta^2 / eta_tilde) + 1)
        - delta^2 = risk_aversion * sigma^2 * tau
        - eta_tilde = eta * (1 - gamma * tau / (2 * eta))
        - tau = 1 / n_periods (time step)
        - eta, gamma derived from volatility and ADV

        Parameters
        ----------
        total_shares : float
            Total number of shares to execute.
        n_periods : int
            Number of trading periods (e.g., time slices).
        volatility : float
            Daily volatility of the asset.
        adv : float
            Average daily volume. Used to calibrate impact
            parameters.
        risk_aversion : float
            Risk aversion parameter. Higher values trade faster
            to reduce timing risk. Default is 1.0.

        Returns
        -------
        List[float]
            Trade schedule: shares to trade in each period
            (n_periods,). Positive values indicate buying;
            the sum equals total_shares.

        Raises
        ------
        ValueError
            If n_periods < 1 or adv <= 0.

        Notes
        -----
        Implements the discrete-time optimal execution model
        from Almgren & Chriss (2001):

        The inventory trajectory n_0, n_1, ..., n_N satisfies:
        - n_0 = X (total shares)
        - n_N = 0 (fully executed)
        - n_j = X * sinh(kappa*(N-j)) / sinh(kappa*N)

        The trade list x_j = n_{j-1} - n_j gives shares traded
        in each period.

        References
        ----------
        Almgren, R. & Chriss, N. (2001). "Optimal execution of
        portfolio transactions." Journal of Risk 3(2):5-39.
        """
        if n_periods < 1:
            raise ValueError(
                f"n_periods must be >= 1, got {n_periods}"
            )
        if adv <= 0:
            raise ValueError(
                f"adv must be positive, got {adv}"
            )

        N = n_periods
        X = total_shares
        sigma = volatility
        tau = 1.0 / N  # time step (fraction of total horizon)

        # Calibrate temporary and permanent impact parameters
        # from market microstructure:
        # eta (temporary impact) ~ sigma * tau / (0.01 * ADV)
        # gamma (permanent impact) ~ 0.1 * eta
        # These are standard calibrations from Almgren et al.
        eta = sigma * tau / (0.01 * adv) if adv > 0 else 1e-6
        gamma = 0.1 * eta

        # Avoid degenerate cases
        eta = max(eta, 1e-14)

        # Adjusted temporary impact
        eta_tilde = eta * (1.0 - gamma * tau / (2.0 * eta))
        eta_tilde = max(eta_tilde, 1e-14)

        # Variance term
        delta_sq = risk_aversion * sigma * sigma * tau

        # kappa: arccosh(1 + delta^2 / (2 * eta_tilde))
        # arccosh(x) = ln(x + sqrt(x^2 - 1)) for x >= 1
        arg = 0.5 * delta_sq / eta_tilde + 1.0
        # arg >= 1 by construction
        if arg <= 1.0:
            kappa = 0.0
        else:
            kappa = math.log(
                arg + math.sqrt(arg * arg - 1.0)
            )

        # Compute inventory trajectory
        # n_j = X * sinh(kappa * (N - j)) / sinh(kappa * N)
        # j = 0, 1, ..., N  with n_0 = X, n_N = 0
        if kappa < 1e-12:
            # Linear execution (TWAP) when kappa -> 0
            inventory = [
                X * (N - j) / N for j in range(N + 1)
            ]
        else:
            sinh_kN = math.sinh(kappa * N)
            if abs(sinh_kN) < 1e-14:
                # Fallback to TWAP
                inventory = [
                    X * (N - j) / N for j in range(N + 1)
                ]
            else:
                inventory = [
                    X * math.sinh(kappa * (N - j)) / sinh_kN
                    for j in range(N + 1)
                ]

        # Trade list: x_j = n_{j-1} - n_j for j = 1..N
        trades = [
            inventory[j - 1] - inventory[j]
            for j in range(1, N + 1)
        ]

        return trades
