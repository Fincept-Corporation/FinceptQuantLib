"""
Mean-Variance Portfolio Optimization

Implements Markowitz (1952) mean-variance optimization:
- Minimum variance portfolio
- Maximum Sharpe ratio portfolio
- Efficient frontier construction
- Target return/risk optimization
- Various constraint types

The optimization problem:
    min  w' Σ w  (portfolio variance)
    s.t. w' μ >= target_return
         w' 1 = 1  (full investment)
         l <= w <= u  (bounds)
         A w >= b  (linear constraints)

References:
- Markowitz, H. (1952). "Portfolio Selection."
- Merton, R.C. (1972). "An Analytic Derivation of the Efficient Portfolio Frontier."
"""

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import math


@dataclass
class PortfolioConstraints:
    """
    Constraints for portfolio optimization.

    Attributes:
        min_weights: Minimum weight per asset (default 0 = long-only)
        max_weights: Maximum weight per asset (default 1)
        group_constraints: Dict of {group_name: (assets, min_weight, max_weight)}
        max_turnover: Maximum turnover from current portfolio
        max_tracking_error: Maximum tracking error vs benchmark
        min_diversification: Minimum diversification ratio
    """
    min_weights: Optional[List[float]] = None
    max_weights: Optional[List[float]] = None
    group_constraints: Optional[Dict[str, Tuple[List[int], float, float]]] = None
    max_turnover: Optional[float] = None
    max_tracking_error: Optional[float] = None
    min_diversification: Optional[float] = None

    def get_bounds(self, n_assets: int) -> List[Tuple[float, float]]:
        """Get bounds for each asset."""
        min_w = self.min_weights or [0.0] * n_assets
        max_w = self.max_weights or [1.0] * n_assets
        return list(zip(min_w, max_w))


@dataclass
class OptimizationResult:
    """
    Result of portfolio optimization.

    Attributes:
        weights: Optimal portfolio weights
        expected_return: Portfolio expected return
        volatility: Portfolio volatility (std dev)
        sharpe_ratio: Sharpe ratio (if risk-free rate provided)
        risk_contributions: Risk contribution per asset
        success: Whether optimization succeeded
        message: Status message
    """
    weights: List[float]
    expected_return: float
    volatility: float
    sharpe_ratio: float = 0.0
    risk_contributions: Optional[List[float]] = None
    success: bool = True
    message: str = "Optimization successful"


class MeanVarianceOptimizer:
    """
    Mean-Variance Portfolio Optimizer.

    Implements Markowitz optimization with various objectives:
    - Minimum variance
    - Maximum Sharpe ratio
    - Target return
    - Target risk

    Uses analytical solution when possible, numerical otherwise.

    Example:
        optimizer = MeanVarianceOptimizer(returns, cov_matrix)
        result = optimizer.optimize(objective='max_sharpe', rf_rate=0.02)
        print(result.weights, result.sharpe_ratio)
    """

    def __init__(
        self,
        expected_returns: List[float],
        covariance_matrix: List[List[float]],
        asset_names: Optional[List[str]] = None
    ):
        """
        Initialize optimizer.

        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            asset_names: Optional names for assets
        """
        self.mu = expected_returns
        self.sigma = covariance_matrix
        self.n_assets = len(expected_returns)
        self.asset_names = asset_names or [f"Asset_{i}" for i in range(self.n_assets)]

        # Validate dimensions
        if len(covariance_matrix) != self.n_assets:
            raise ValueError("Covariance matrix dimensions don't match returns")
        for i, row in enumerate(covariance_matrix):
            if len(row) != self.n_assets:
                raise ValueError(f"Covariance matrix row {i} has {len(row)} columns, expected {self.n_assets}")

        # Pre-compute inverse covariance (for analytical solutions)
        self._sigma_inv = self._matrix_inverse(covariance_matrix)

    def min_variance(
        self,
        constraints: Optional[PortfolioConstraints] = None
    ) -> OptimizationResult:
        """
        Find the minimum variance portfolio.

        Analytical solution (unconstrained):
        w* = Σ⁻¹ 1 / (1' Σ⁻¹ 1)
        """
        if constraints is None:
            # Analytical solution
            ones = [1.0] * self.n_assets
            sigma_inv_ones = self._matrix_vector_mult(self._sigma_inv, ones)
            denom = sum(sigma_inv_ones)

            weights = [x / denom for x in sigma_inv_ones]
        else:
            # Numerical optimization
            weights = self._optimize_numerical(
                objective='min_variance',
                constraints=constraints
            )

        return self._build_result(weights)

    def max_sharpe(
        self,
        rf_rate: float = 0.0,
        constraints: Optional[PortfolioConstraints] = None
    ) -> OptimizationResult:
        """
        Find the maximum Sharpe ratio portfolio.

        Analytical solution (unconstrained):
        w* = Σ⁻¹ (μ - rf) / (1' Σ⁻¹ (μ - rf))
        """
        if constraints is None:
            # Analytical solution
            excess_returns = [mu - rf_rate for mu in self.mu]
            sigma_inv_excess = self._matrix_vector_mult(self._sigma_inv, excess_returns)
            denom = sum(sigma_inv_excess)

            if abs(denom) < 1e-10:
                return self.min_variance(constraints)

            weights = [x / denom for x in sigma_inv_excess]

            # Handle negative weights (means rf_rate is too high)
            if min(weights) < -1e-10:
                weights = self._optimize_numerical(
                    objective='max_sharpe',
                    constraints=PortfolioConstraints(min_weights=[0.0] * self.n_assets),
                    rf_rate=rf_rate
                )
        else:
            weights = self._optimize_numerical(
                objective='max_sharpe',
                constraints=constraints,
                rf_rate=rf_rate
            )

        result = self._build_result(weights, rf_rate)
        return result

    def target_return(
        self,
        target: float,
        constraints: Optional[PortfolioConstraints] = None
    ) -> OptimizationResult:
        """
        Find minimum variance portfolio achieving target return.

        Analytical solution uses Lagrange multipliers.
        """
        if constraints is None:
            # Analytical solution using two-fund theorem
            ones = [1.0] * self.n_assets

            # Compute key quantities
            A = self._quadratic_form(self.mu, self._sigma_inv, self.mu)
            B = self._quadratic_form(ones, self._sigma_inv, self.mu)
            C = self._quadratic_form(ones, self._sigma_inv, ones)

            D = A * C - B * B

            if abs(D) < 1e-10:
                return self.min_variance(constraints)

            # Optimal weights
            lambda1 = (C * target - B) / D
            lambda2 = (A - B * target) / D

            sigma_inv_mu = self._matrix_vector_mult(self._sigma_inv, self.mu)
            sigma_inv_ones = self._matrix_vector_mult(self._sigma_inv, ones)

            weights = [
                lambda1 * sigma_inv_mu[i] + lambda2 * sigma_inv_ones[i]
                for i in range(self.n_assets)
            ]
        else:
            weights = self._optimize_numerical(
                objective='target_return',
                constraints=constraints,
                target_return=target
            )

        return self._build_result(weights)

    def target_risk(
        self,
        target: float,
        constraints: Optional[PortfolioConstraints] = None
    ) -> OptimizationResult:
        """
        Find maximum return portfolio achieving target volatility.
        """
        weights = self._optimize_numerical(
            objective='target_risk',
            constraints=constraints,
            target_risk=target
        )
        return self._build_result(weights)

    def efficient_frontier(
        self,
        n_points: int = 50,
        constraints: Optional[PortfolioConstraints] = None
    ) -> List[OptimizationResult]:
        """
        Compute the efficient frontier.

        Returns a list of optimal portfolios from min variance to max return.
        """
        if n_points < 2:
            return [self.min_variance(constraints)]

        # Find endpoints
        min_var = self.min_variance(constraints)
        max_ret = max(self.mu)

        # Generate target returns
        targets = []
        for i in range(n_points):
            t = min_var.expected_return + (max_ret - min_var.expected_return) * i / (n_points - 1)
            targets.append(t)

        # Compute efficient portfolios
        frontier = []
        for target in targets:
            try:
                result = self.target_return(target, constraints)
                if result.success:
                    frontier.append(result)
            except Exception:
                continue

        return frontier

    def _optimize_numerical(
        self,
        objective: str,
        constraints: Optional[PortfolioConstraints] = None,
        rf_rate: float = 0.0,
        target_return: Optional[float] = None,
        target_risk: Optional[float] = None
    ) -> List[float]:
        """
        Numerical optimization using gradient descent.

        Simple implementation; production would use scipy.optimize.
        """
        # Initialize with equal weights
        weights = [1.0 / self.n_assets] * self.n_assets
        bounds = constraints.get_bounds(self.n_assets) if constraints else [(0, 1)] * self.n_assets

        learning_rate = 0.1
        n_iterations = 1000

        for iteration in range(n_iterations):
            # Compute gradient based on objective
            if objective == 'min_variance':
                grad = self._variance_gradient(weights)
            elif objective == 'max_sharpe':
                grad = self._sharpe_gradient(weights, rf_rate)
                grad = [-g for g in grad]  # Maximize
            elif objective == 'target_return':
                grad = self._variance_gradient(weights)
                # Add penalty for missing target
                if target_return is None:
                    raise ValueError("target_return is required for 'target_return' objective")
                current_ret = sum(w * mu for w, mu in zip(weights, self.mu))
                penalty_grad = [10 * (current_ret - target_return) * mu for mu in self.mu]
                grad = [g + p for g, p in zip(grad, penalty_grad)]
            elif objective == 'target_risk':
                if target_risk is None:
                    raise ValueError("target_risk is required for 'target_risk' objective")
                grad = [-mu for mu in self.mu]  # Maximize return
                # Penalty for missing target risk
                current_risk = math.sqrt(max(0.0, self._portfolio_variance(weights)))
                if current_risk > target_risk:
                    var_grad = self._variance_gradient(weights)
                    grad = [g + 10 * (current_risk - target_risk) * vg for g, vg in zip(grad, var_grad)]

            # Update weights
            for i in range(self.n_assets):
                weights[i] -= learning_rate * grad[i]
                weights[i] = max(bounds[i][0], min(bounds[i][1], weights[i]))

            # Normalize to sum to 1, then re-clip to bounds
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
                weights = [max(bounds[i][0], min(bounds[i][1], weights[i]))
                           for i in range(self.n_assets)]

            # Reduce learning rate
            learning_rate *= 0.999

        return weights

    def _variance_gradient(self, weights: List[float]) -> List[float]:
        """Gradient of portfolio variance: 2 Σ w"""
        return [2 * sum(self.sigma[i][j] * weights[j] for j in range(self.n_assets))
                for i in range(self.n_assets)]

    def _sharpe_gradient(self, weights: List[float], rf_rate: float) -> List[float]:
        """Gradient of Sharpe ratio."""
        ret = sum(w * mu for w, mu in zip(weights, self.mu))
        var = self._portfolio_variance(weights)
        vol = math.sqrt(var) if var > 0 else 1e-10
        excess = ret - rf_rate

        var_grad = self._variance_gradient(weights)

        grad = []
        for i in range(self.n_assets):
            # d(Sharpe)/dw = (μ/σ) - (excess_ret / (2σ³)) × dVar/dw
            g = self.mu[i] / vol - excess * var_grad[i] / (2 * vol ** 3)
            grad.append(g)

        return grad

    def _portfolio_variance(self, weights: List[float]) -> float:
        """Calculate portfolio variance: w' Σ w"""
        var = 0.0
        for i in range(self.n_assets):
            for j in range(self.n_assets):
                var += weights[i] * weights[j] * self.sigma[i][j]
        return var

    def _build_result(self, weights: List[float], rf_rate: float = 0.0) -> OptimizationResult:
        """Build result object from weights."""
        expected_return = sum(w * mu for w, mu in zip(weights, self.mu))
        variance = self._portfolio_variance(weights)
        volatility = math.sqrt(variance) if variance > 0 else 0

        sharpe = (expected_return - rf_rate) / volatility if volatility > 0 else 0

        # Risk contributions
        risk_contributions = self._risk_contributions(weights)

        return OptimizationResult(
            weights=weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            risk_contributions=risk_contributions
        )

    def _risk_contributions(self, weights: List[float]) -> List[float]:
        """
        Calculate risk contribution per asset.

        RC_i = w_i × (Σw)_i / σ_p
        """
        sigma_w = self._matrix_vector_mult(self.sigma, weights)
        var = self._portfolio_variance(weights)
        vol = math.sqrt(var) if var > 0 else 1

        return [weights[i] * sigma_w[i] / vol for i in range(self.n_assets)]

    def _matrix_inverse(self, matrix: List[List[float]]) -> List[List[float]]:
        """Simple matrix inverse using Gauss-Jordan elimination."""
        n = len(matrix)
        # Augment with identity
        aug = [[matrix[i][j] if j < n else float(i == j - n)
                for j in range(2 * n)] for i in range(n)]

        for col in range(n):
            # Find pivot
            max_row = col
            for row in range(col + 1, n):
                if abs(aug[row][col]) > abs(aug[max_row][col]):
                    max_row = row
            aug[col], aug[max_row] = aug[max_row], aug[col]

            pivot = aug[col][col]
            if abs(pivot) < 1e-10:
                raise ValueError("Singular covariance matrix — cannot compute inverse")

            # Scale row
            for j in range(2 * n):
                aug[col][j] /= pivot

            # Eliminate
            for row in range(n):
                if row != col:
                    factor = aug[row][col]
                    for j in range(2 * n):
                        aug[row][j] -= factor * aug[col][j]

        return [[aug[i][j] for j in range(n, 2 * n)] for i in range(n)]

    def _matrix_vector_mult(
        self,
        matrix: List[List[float]],
        vector: List[float]
    ) -> List[float]:
        """Matrix-vector multiplication."""
        n = len(matrix)
        return [sum(matrix[i][j] * vector[j] for j in range(n)) for i in range(n)]

    def _quadratic_form(
        self,
        v1: List[float],
        matrix: List[List[float]],
        v2: List[float]
    ) -> float:
        """Compute v1' M v2."""
        Mv2 = self._matrix_vector_mult(matrix, v2)
        return sum(v1[i] * Mv2[i] for i in range(len(v1)))


# Convenience functions

def efficient_frontier(
    expected_returns: List[float],
    covariance_matrix: List[List[float]],
    n_points: int = 50,
    constraints: Optional[PortfolioConstraints] = None
) -> List[OptimizationResult]:
    """Compute efficient frontier."""
    optimizer = MeanVarianceOptimizer(expected_returns, covariance_matrix)
    return optimizer.efficient_frontier(n_points, constraints)


def min_variance_portfolio(
    expected_returns: List[float],
    covariance_matrix: List[List[float]],
    constraints: Optional[PortfolioConstraints] = None
) -> OptimizationResult:
    """Find minimum variance portfolio."""
    optimizer = MeanVarianceOptimizer(expected_returns, covariance_matrix)
    return optimizer.min_variance(constraints)


def max_sharpe_portfolio(
    expected_returns: List[float],
    covariance_matrix: List[List[float]],
    rf_rate: float = 0.0,
    constraints: Optional[PortfolioConstraints] = None
) -> OptimizationResult:
    """Find maximum Sharpe ratio portfolio."""
    optimizer = MeanVarianceOptimizer(expected_returns, covariance_matrix)
    return optimizer.max_sharpe(rf_rate, constraints)


def target_return_portfolio(
    expected_returns: List[float],
    covariance_matrix: List[List[float]],
    target: float,
    constraints: Optional[PortfolioConstraints] = None
) -> OptimizationResult:
    """Find minimum variance portfolio achieving target return."""
    optimizer = MeanVarianceOptimizer(expected_returns, covariance_matrix)
    return optimizer.target_return(target, constraints)
