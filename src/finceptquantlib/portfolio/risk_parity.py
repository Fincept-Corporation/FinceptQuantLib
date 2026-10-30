"""
Risk Parity and Risk-Based Portfolio Construction

Implements risk-based allocation strategies:
- Risk Parity / Equal Risk Contribution (ERC)
- Hierarchical Risk Parity (HRP)
- Maximum Diversification
- Inverse Volatility Weighting

Risk Parity objective: Equal risk contribution from each asset
    RC_i = w_i × (Σw)_i = σ_p / N  for all i

References:
- Maillard, Roncalli, Teiletche (2010). "On the Properties of Equally-Weighted Risk Contribution Portfolios."
- Qian (2005). "Risk Parity Portfolios."
- Lopez de Prado (2016). "Building Diversified Portfolios that Outperform Out-of-Sample."
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import math


@dataclass
class RiskParityResult:
    """
    Result of risk parity optimization.

    Attributes:
        weights: Portfolio weights
        risk_contributions: Risk contribution per asset
        total_risk: Portfolio volatility
        diversification_ratio: Diversification ratio
        effective_n: Effective number of bets
    """
    weights: List[float]
    risk_contributions: List[float]
    total_risk: float
    diversification_ratio: float = 1.0
    effective_n: float = 0.0


class RiskParity:
    """
    Risk Parity Portfolio Construction.

    Allocates to equalize risk contribution from each asset.
    The risk contribution of asset i is:
        RC_i = w_i × (Σw)_i / σ_p

    Where σ_p = √(w'Σw) is portfolio volatility.

    The optimization finds weights such that RC_i = RC_j for all i,j.
    """

    def __init__(self, covariance_matrix: List[List[float]]):
        """
        Initialize risk parity optimizer.

        Args:
            covariance_matrix: Asset covariance matrix
        """
        self.sigma = covariance_matrix
        self.n_assets = len(covariance_matrix)

    def optimize(
        self,
        target_risk_contributions: Optional[List[float]] = None,
        max_iterations: int = 1000,
        tolerance: float = 1e-10
    ) -> RiskParityResult:
        """
        Find risk parity portfolio.

        Uses Spinu (2013) Newton method for ERC optimization.

        Args:
            target_risk_contributions: Target RC per asset (default = equal)
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            RiskParityResult with optimal weights
        """
        if target_risk_contributions is None:
            target_risk_contributions = [1.0 / self.n_assets] * self.n_assets

        # Normalize targets
        total = sum(target_risk_contributions)
        budgets = [t / total for t in target_risk_contributions]

        # Initialize with inverse volatility
        weights = self._inverse_volatility_weights()

        # Newton iteration for ERC
        for iteration in range(max_iterations):
            # Current risk metrics
            sigma_w = self._matrix_vector_mult(self.sigma, weights)
            var = sum(weights[i] * sigma_w[i] for i in range(self.n_assets))
            vol = math.sqrt(var) if var > 0 else 1e-10

            # Risk contributions (as fraction of total)
            mrc = [sigma_w[i] / vol for i in range(self.n_assets)]
            rc = [weights[i] * mrc[i] / vol for i in range(self.n_assets)]

            # Check convergence
            max_diff = max(abs(rc[i] - budgets[i]) for i in range(self.n_assets))
            if max_diff < tolerance:
                break

            # Newton step
            # Gradient of risk contributions
            for i in range(self.n_assets):
                # Adjust weight to match target RC
                if mrc[i] > 1e-10:
                    target_weight = budgets[i] * vol / mrc[i]
                    # Smooth adjustment
                    weights[i] = 0.9 * weights[i] + 0.1 * target_weight

            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]

        # Final metrics
        return self._build_result(weights)

    def _inverse_volatility_weights(self) -> List[float]:
        """Initialize with inverse volatility weights."""
        vols = [math.sqrt(self.sigma[i][i]) for i in range(self.n_assets)]
        inv_vols = [1.0 / v if v > 0 else 1.0 for v in vols]
        total = sum(inv_vols)
        return [iv / total for iv in inv_vols]

    def _build_result(self, weights: List[float]) -> RiskParityResult:
        """Build result from weights."""
        sigma_w = self._matrix_vector_mult(self.sigma, weights)
        var = sum(weights[i] * sigma_w[i] for i in range(self.n_assets))
        vol = math.sqrt(var) if var > 0 else 0

        # Risk contributions
        if vol > 0:
            mrc = [sigma_w[i] / vol for i in range(self.n_assets)]
            rc = [weights[i] * mrc[i] for i in range(self.n_assets)]
        else:
            rc = [0.0] * self.n_assets

        # Diversification ratio: sum of weighted vols / portfolio vol
        weighted_vols = sum(
            weights[i] * math.sqrt(self.sigma[i][i])
            for i in range(self.n_assets)
        )
        div_ratio = weighted_vols / vol if vol > 0 else 1.0

        # Effective N: 1 / sum(rc_frac_i^2) where rc_frac sums to 1
        rc_total = sum(rc)
        if rc_total > 0:
            rc_frac = [r / rc_total for r in rc]
            rc_sq = sum(r ** 2 for r in rc_frac)
            eff_n = 1.0 / rc_sq if rc_sq > 0 else self.n_assets
        else:
            eff_n = float(self.n_assets)

        return RiskParityResult(
            weights=weights,
            risk_contributions=rc,
            total_risk=vol,
            diversification_ratio=div_ratio,
            effective_n=eff_n
        )

    def _matrix_vector_mult(
        self,
        matrix: List[List[float]],
        vector: List[float]
    ) -> List[float]:
        n = len(matrix)
        return [sum(matrix[i][j] * vector[j] for j in range(n)) for i in range(n)]


class EqualRiskContribution(RiskParity):
    """
    Equal Risk Contribution (ERC) Portfolio.

    Special case of risk parity where each asset contributes
    equally to portfolio risk.

    This is equivalent to RiskParity with equal budgets.
    """

    def optimize(
        self,
        target_risk_contributions: Optional[List[float]] = None,
        max_iterations: int = 1000,
        tolerance: float = 1e-10,
    ) -> RiskParityResult:
        """Find ERC portfolio (equal budgets, ignores target_risk_contributions)."""
        return super().optimize(
            target_risk_contributions=[1.0 / self.n_assets] * self.n_assets,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )


class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity (HRP).

    Lopez de Prado (2016) algorithm:
    1. Compute correlation distance matrix
    2. Cluster assets using hierarchical clustering
    3. Quasi-diagonalize covariance matrix
    4. Recursive bisection for allocation

    Key advantages:
    - No matrix inversion required
    - More stable out-of-sample
    - Accounts for correlation structure
    """

    def __init__(
        self,
        covariance_matrix: List[List[float]],
        correlation_matrix: Optional[List[List[float]]] = None
    ):
        """
        Initialize HRP.

        Args:
            covariance_matrix: Asset covariance matrix
            correlation_matrix: Optional correlation matrix (computed if not provided)
        """
        self.sigma = covariance_matrix
        self.n_assets = len(covariance_matrix)

        if correlation_matrix is None:
            self.corr = self._cov_to_corr(covariance_matrix)
        else:
            self.corr = correlation_matrix

    def optimize(self) -> RiskParityResult:
        """
        Compute HRP weights.

        Returns:
            RiskParityResult with HRP weights
        """
        # 1. Compute distance matrix
        dist = self._correlation_distance()

        # 2. Hierarchical clustering (single linkage)
        linkage = self._hierarchical_cluster(dist)

        # 3. Get ordering from dendrogram
        order = self._get_quasi_diagonal_order(linkage)

        # 4. Recursive bisection
        weights = self._recursive_bisection(order)

        # Build result
        return self._build_result(weights)

    def _correlation_distance(self) -> List[List[float]]:
        """
        Compute correlation-based distance matrix.

        d_ij = √(0.5 × (1 - ρ_ij))
        """
        dist = [[0.0] * self.n_assets for _ in range(self.n_assets)]
        for i in range(self.n_assets):
            for j in range(self.n_assets):
                dist[i][j] = math.sqrt(0.5 * (1 - self.corr[i][j]))
        return dist

    def _hierarchical_cluster(self, dist: List[List[float]]) -> List[Tuple[int, int, float, int]]:
        """
        Simple single-linkage hierarchical clustering.

        Returns linkage matrix: list of (cluster1, cluster2, distance, size)
        """
        n = self.n_assets
        # Initialize clusters
        clusters = {i: [i] for i in range(n)}
        cluster_id = n

        linkage = []

        while len(clusters) > 1:
            # Find closest pair
            min_dist = float('inf')
            merge_i, merge_j = -1, -1

            cluster_ids = list(clusters.keys())
            for i_idx, ci in enumerate(cluster_ids):
                for cj in cluster_ids[i_idx + 1:]:
                    # Single linkage: min distance between any pair
                    d = self._cluster_distance(clusters[ci], clusters[cj], dist)
                    if d < min_dist:
                        min_dist = d
                        merge_i, merge_j = ci, cj

            # Merge clusters
            new_cluster = clusters[merge_i] + clusters[merge_j]
            linkage.append((merge_i, merge_j, min_dist, len(new_cluster)))

            del clusters[merge_i]
            del clusters[merge_j]
            clusters[cluster_id] = new_cluster
            cluster_id += 1

        return linkage

    def _cluster_distance(
        self,
        c1: List[int],
        c2: List[int],
        dist: List[List[float]]
    ) -> float:
        """Single linkage: minimum distance between clusters."""
        min_d = float('inf')
        for i in c1:
            for j in c2:
                min_d = min(min_d, dist[i][j])
        return min_d

    def _get_quasi_diagonal_order(self, linkage: List[Tuple[int, int, float, int]]) -> List[int]:
        """
        Get asset ordering that quasi-diagonalizes covariance.

        Traverses dendrogram to get leaf order.
        """
        n = self.n_assets
        # Build tree structure
        tree = {i: [i] for i in range(n)}
        node_id = n

        for merge_i, merge_j, _, _ in linkage:
            tree[node_id] = tree[merge_i] + tree[merge_j]
            node_id += 1

        # The last node contains all assets in order
        return tree[node_id - 1]

    def _recursive_bisection(self, order: List[int]) -> List[float]:
        """
        Recursive bisection for weight allocation.

        Splits the ordered assets recursively, allocating based on
        cluster variance.
        """
        weights = [1.0] * self.n_assets

        def bisect(indices: List[int], w: float):
            if len(indices) == 1:
                weights[indices[0]] = w
                return

            # Split in half
            mid = len(indices) // 2
            left = indices[:mid]
            right = indices[mid:]

            # Compute cluster variances
            var_left = self._cluster_variance(left)
            var_right = self._cluster_variance(right)

            # Allocate inverse to variance
            total_var = var_left + var_right
            if total_var > 0:
                alpha = 1 - var_left / total_var
            else:
                alpha = 0.5

            bisect(left, w * alpha)
            bisect(right, w * (1 - alpha))

        bisect(order, 1.0)

        return weights

    def _cluster_variance(self, indices: List[int]) -> float:
        """
        Compute variance of inverse-volatility weighted cluster.
        """
        if len(indices) == 1:
            return self.sigma[indices[0]][indices[0]]

        # Inverse volatility weights within cluster
        vols = [math.sqrt(self.sigma[i][i]) for i in indices]
        inv_vols = [1.0 / v if v > 0 else 1.0 for v in vols]
        total = sum(inv_vols)
        w = [iv / total for iv in inv_vols]

        # Cluster variance
        var = 0.0
        for i_idx, i in enumerate(indices):
            for j_idx, j in enumerate(indices):
                var += w[i_idx] * w[j_idx] * self.sigma[i][j]

        return var

    def _cov_to_corr(self, cov: List[List[float]]) -> List[List[float]]:
        """Convert covariance to correlation matrix."""
        n = len(cov)
        stds = [math.sqrt(cov[i][i]) if cov[i][i] > 0 else 1.0 for i in range(n)]
        return [[cov[i][j] / (stds[i] * stds[j]) for j in range(n)] for i in range(n)]

    def _build_result(self, weights: List[float]) -> RiskParityResult:
        """Build result from weights."""
        sigma_w = [sum(self.sigma[i][j] * weights[j] for j in range(self.n_assets))
                   for i in range(self.n_assets)]
        var = sum(weights[i] * sigma_w[i] for i in range(self.n_assets))
        vol = math.sqrt(var) if var > 0 else 0

        if vol > 0:
            rc = [weights[i] * sigma_w[i] / vol for i in range(self.n_assets)]
        else:
            rc = [0.0] * self.n_assets

        weighted_vols = sum(
            weights[i] * math.sqrt(self.sigma[i][i])
            for i in range(self.n_assets)
        )
        div_ratio = weighted_vols / vol if vol > 0 else 1.0

        # Effective N: 1 / sum(rc_frac_i^2) where rc_frac sums to 1
        rc_total = sum(rc)
        if rc_total > 0:
            rc_frac = [r / rc_total for r in rc]
            rc_sq = sum(r ** 2 for r in rc_frac)
            eff_n = 1.0 / rc_sq if rc_sq > 0 else self.n_assets
        else:
            eff_n = float(self.n_assets)

        return RiskParityResult(
            weights=weights,
            risk_contributions=rc,
            total_risk=vol,
            diversification_ratio=div_ratio,
            effective_n=eff_n
        )


# Convenience functions

def risk_parity_weights(covariance_matrix: List[List[float]]) -> List[float]:
    """Compute equal risk contribution weights."""
    optimizer = EqualRiskContribution(covariance_matrix)
    result = optimizer.optimize()
    return result.weights


def hrp_weights(covariance_matrix: List[List[float]]) -> List[float]:
    """Compute Hierarchical Risk Parity weights."""
    optimizer = HierarchicalRiskParity(covariance_matrix)
    result = optimizer.optimize()
    return result.weights


def inverse_volatility_weights(covariance_matrix: List[List[float]]) -> List[float]:
    """Compute inverse volatility weights."""
    n = len(covariance_matrix)
    vols = [math.sqrt(covariance_matrix[i][i]) for i in range(n)]
    inv_vols = [1.0 / v if v > 0 else 1.0 for v in vols]
    total = sum(inv_vols)
    return [iv / total for iv in inv_vols]
