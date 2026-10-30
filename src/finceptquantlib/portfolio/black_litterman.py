"""
Black-Litterman Model

Implements the Black-Litterman (1992) approach to portfolio construction:
- Combines equilibrium returns with investor views
- Provides a Bayesian framework for blending prior and posterior
- Reduces estimation error impact vs pure mean-variance

The model:
    E[r] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ × [(τΣ)⁻¹π + P'Ω⁻¹Q]

where:
    π: equilibrium expected returns (from CAPM)
    P: view pick matrix (what assets involved in views)
    Q: view expected returns
    Ω: view uncertainty matrix
    τ: scalar (typically 0.025-0.05)
    Σ: covariance matrix

References:
- Black, F. & Litterman, R. (1992). "Global Portfolio Optimization."
- He, G. & Litterman, R. (1999). "The Intuition Behind Black-Litterman."
- Idzorek, T. (2007). "A Step-by-Step Guide to the Black-Litterman Model."
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum


class ViewType(Enum):
    """Type of investor view."""
    ABSOLUTE = "absolute"  # Asset X will return Y%
    RELATIVE = "relative"  # Asset X will outperform Y by Z%


@dataclass
class View:
    """
    An investor view on expected returns.

    For absolute view: "Asset A will return 5%"
        assets = [A]
        weights = [1.0]
        expected_return = 0.05

    For relative view: "Asset A will outperform B by 3%"
        assets = [A, B]
        weights = [1.0, -1.0]
        expected_return = 0.03

    Attributes:
        assets: List of asset indices involved
        weights: Weights for each asset (+1 for long, -1 for short)
        expected_return: Expected return of the view
        confidence: View confidence (0-1), higher = more certain
        view_type: ABSOLUTE or RELATIVE
    """
    assets: List[int]
    weights: List[float]
    expected_return: float
    confidence: float = 0.5
    view_type: ViewType = ViewType.ABSOLUTE

    def __post_init__(self):
        if len(self.assets) != len(self.weights):
            raise ValueError("Assets and weights must have same length")
        if not (0 < self.confidence <= 1):
            raise ValueError("Confidence must be in (0, 1]")

        # Relative views: weights should sum to zero (e.g. [1, -1])
        # Do NOT normalize by sum(abs(w)) — that destroys the pick matrix semantics
        if self.view_type == ViewType.RELATIVE:
            if abs(sum(self.weights)) > 1e-10:
                raise ValueError("Relative view weights must sum to zero")


class BlackLitterman:
    """
    Black-Litterman Model for portfolio optimization.

    Combines market equilibrium with investor views to produce
    posterior expected returns that can be used in mean-variance
    optimization.

    Example:
        bl = BlackLitterman(
            covariance_matrix=cov,
            market_caps=caps,
            risk_aversion=2.5
        )
        bl.add_absolute_view(0, 0.10, confidence=0.8)
        bl.add_relative_view(1, 2, 0.02, confidence=0.6)
        posterior_returns = bl.posterior_returns()
    """

    def __init__(
        self,
        covariance_matrix: List[List[float]],
        market_caps: Optional[List[float]] = None,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        rf_rate: float = 0.0
    ):
        """
        Initialize Black-Litterman model.

        Args:
            covariance_matrix: Asset covariance matrix
            market_caps: Market capitalizations (for equilibrium weights)
            risk_aversion: Risk aversion parameter δ
            tau: Scalar for prior uncertainty (typically 0.025-0.05)
            rf_rate: Risk-free rate
        """
        self.sigma = covariance_matrix
        self.n_assets = len(covariance_matrix)
        self.delta = risk_aversion
        self.tau = tau
        self.rf_rate = rf_rate

        # Market cap weights
        if market_caps is None:
            self.market_weights = [1.0 / self.n_assets] * self.n_assets
        else:
            total = sum(market_caps)
            if total <= 0:
                raise ValueError("Total market capitalization must be positive")
            self.market_weights = [cap / total for cap in market_caps]

        # Compute equilibrium returns
        self.equilibrium_returns = self._compute_equilibrium()

        # Views storage
        self.views: List[View] = []

    def _compute_equilibrium(self) -> List[float]:
        """
        Compute equilibrium (CAPM implied) returns.

        π = δ × Σ × w_mkt

        where w_mkt are market cap weights.
        """
        sigma_w = self._matrix_vector_mult(self.sigma, self.market_weights)
        return [self.delta * sw for sw in sigma_w]

    def add_view(self, view: View):
        """Add a view to the model."""
        # Validate assets
        for asset in view.assets:
            if not (0 <= asset < self.n_assets):
                raise ValueError(f"Asset index {asset} out of range")
        self.views.append(view)

    def add_absolute_view(
        self,
        asset: int,
        expected_return: float,
        confidence: float = 0.5
    ):
        """
        Add an absolute view on a single asset.

        "Asset i will return expected_return"
        """
        view = View(
            assets=[asset],
            weights=[1.0],
            expected_return=expected_return,
            confidence=confidence,
            view_type=ViewType.ABSOLUTE
        )
        self.add_view(view)

    def add_relative_view(
        self,
        asset_long: int,
        asset_short: int,
        outperformance: float,
        confidence: float = 0.5
    ):
        """
        Add a relative view.

        "Asset_long will outperform asset_short by outperformance"
        """
        view = View(
            assets=[asset_long, asset_short],
            weights=[1.0, -1.0],
            expected_return=outperformance,
            confidence=confidence,
            view_type=ViewType.RELATIVE
        )
        self.add_view(view)

    def clear_views(self):
        """Remove all views."""
        self.views = []

    def _build_view_matrices(self) -> Tuple[List[List[float]], List[float], List[List[float]]]:
        """
        Build P, Q, Ω matrices from views.

        P: K × N pick matrix
        Q: K × 1 view returns
        Ω: K × K view uncertainty (diagonal)
        """
        K = len(self.views)
        N = self.n_assets

        # Pick matrix P
        P = [[0.0] * N for _ in range(K)]
        for k, view in enumerate(self.views):
            for asset, weight in zip(view.assets, view.weights):
                P[k][asset] = weight

        # View returns Q
        Q = [view.expected_return for view in self.views]

        # View uncertainty Ω (Idzorek's method based on confidence)
        # Higher confidence → lower variance
        Omega = [[0.0] * K for _ in range(K)]
        for k, view in enumerate(self.views):
            # Variance of view k: P_k × (τΣ) × P_k' scaled by (1-c)/c
            P_k = P[k]
            tau_sigma_Pk = self._matrix_vector_mult(
                [[self.tau * self.sigma[i][j] for j in range(N)] for i in range(N)],
                P_k
            )
            view_var = sum(P_k[i] * tau_sigma_Pk[i] for i in range(N))

            # Scale by confidence (Idzorek adjustment)
            # confidence=1 → omega=0 (perfect certainty)
            # confidence=0.5 → omega = view_var
            c = view.confidence
            if c >= 0.999:
                omega_k = 1e-10
            else:
                omega_k = view_var * (1 - c) / c

            Omega[k][k] = omega_k

        return P, Q, Omega

    def posterior_returns(self) -> List[float]:
        """
        Compute posterior expected returns.

        E[r] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ × [(τΣ)⁻¹π + P'Ω⁻¹Q]

        If no views, returns equilibrium returns.
        """
        if not self.views:
            return self.equilibrium_returns.copy()

        P, Q, Omega = self._build_view_matrices()
        K = len(self.views)
        N = self.n_assets

        # τΣ
        tau_sigma = [[self.tau * self.sigma[i][j] for j in range(N)] for i in range(N)]
        tau_sigma_inv = self._matrix_inverse(tau_sigma)

        # Ω⁻¹
        omega_inv = [[0.0] * K for _ in range(K)]
        for k in range(K):
            if abs(Omega[k][k]) > 1e-15:
                omega_inv[k][k] = 1.0 / Omega[k][k]
            else:
                omega_inv[k][k] = 1e10  # Very certain

        # P' Ω⁻¹ P (N × N)
        Pt_omega_inv = self._matrix_mult(self._transpose(P), omega_inv)
        Pt_omega_inv_P = self._matrix_mult(Pt_omega_inv, P)

        # (τΣ)⁻¹ + P'Ω⁻¹P
        M_inv = [[tau_sigma_inv[i][j] + Pt_omega_inv_P[i][j]
                  for j in range(N)] for i in range(N)]

        # Invert to get posterior covariance factor
        M = self._matrix_inverse(M_inv)

        # (τΣ)⁻¹ π
        tau_sigma_inv_pi = self._matrix_vector_mult(tau_sigma_inv, self.equilibrium_returns)

        # P' Ω⁻¹ Q
        omega_inv_Q = self._matrix_vector_mult(omega_inv, Q)
        Pt_omega_inv_Q = self._matrix_vector_mult(self._transpose(P), omega_inv_Q)

        # Combined: (τΣ)⁻¹π + P'Ω⁻¹Q
        combined = [tau_sigma_inv_pi[i] + Pt_omega_inv_Q[i] for i in range(N)]

        # Posterior returns: M × combined
        posterior = self._matrix_vector_mult(M, combined)

        return posterior

    def posterior_covariance(self) -> List[List[float]]:
        """
        Compute posterior covariance matrix.

        Σ_posterior = Σ + [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹
        """
        if not self.views:
            # Return original with slight shrinkage
            return [[self.sigma[i][j] * (1 + self.tau)
                     for j in range(self.n_assets)]
                    for i in range(self.n_assets)]

        P, Q, Omega = self._build_view_matrices()
        K = len(self.views)
        N = self.n_assets

        tau_sigma = [[self.tau * self.sigma[i][j] for j in range(N)] for i in range(N)]
        tau_sigma_inv = self._matrix_inverse(tau_sigma)

        omega_inv = [[0.0] * K for _ in range(K)]
        for k in range(K):
            omega_inv[k][k] = 1.0 / Omega[k][k] if Omega[k][k] > 1e-15 else 1e10

        Pt_omega_inv = self._matrix_mult(self._transpose(P), omega_inv)
        Pt_omega_inv_P = self._matrix_mult(Pt_omega_inv, P)

        M_inv = [[tau_sigma_inv[i][j] + Pt_omega_inv_P[i][j]
                  for j in range(N)] for i in range(N)]
        M = self._matrix_inverse(M_inv)

        # Posterior = Σ + M
        posterior_cov = [[self.sigma[i][j] + M[i][j]
                          for j in range(N)] for i in range(N)]

        return posterior_cov

    def optimal_weights(self, constraints=None) -> List[float]:
        """
        Compute optimal portfolio weights using posterior returns.

        Uses mean-variance optimization with posterior returns.
        """
        from finceptquantlib.portfolio.optimization import MeanVarianceOptimizer

        posterior = self.posterior_returns()
        posterior_cov = self.posterior_covariance()

        optimizer = MeanVarianceOptimizer(posterior, posterior_cov)
        result = optimizer.max_sharpe(self.rf_rate, constraints)

        return result.weights

    def _matrix_vector_mult(
        self,
        matrix: List[List[float]],
        vector: List[float]
    ) -> List[float]:
        """Matrix-vector multiplication."""
        n = len(matrix)
        m = len(matrix[0]) if n > 0 else 0
        return [sum(matrix[i][j] * vector[j] for j in range(m)) for i in range(n)]

    def _matrix_mult(
        self,
        A: List[List[float]],
        B: List[List[float]]
    ) -> List[List[float]]:
        """Matrix multiplication A × B."""
        n = len(A)
        m = len(B[0]) if B else 0
        k = len(B)
        return [[sum(A[i][l] * B[l][j] for l in range(k)) for j in range(m)] for i in range(n)]

    def _transpose(self, matrix: List[List[float]]) -> List[List[float]]:
        """Matrix transpose."""
        n = len(matrix)
        m = len(matrix[0]) if n > 0 else 0
        return [[matrix[j][i] for j in range(n)] for i in range(m)]

    def _matrix_inverse(self, matrix: List[List[float]]) -> List[List[float]]:
        """Matrix inverse using Gauss-Jordan."""
        n = len(matrix)
        aug = [[matrix[i][j] if j < n else float(i == j - n)
                for j in range(2 * n)] for i in range(n)]

        for col in range(n):
            max_row = col
            for row in range(col + 1, n):
                if abs(aug[row][col]) > abs(aug[max_row][col]):
                    max_row = row
            aug[col], aug[max_row] = aug[max_row], aug[col]

            pivot = aug[col][col]
            if abs(pivot) < 1e-15:
                raise ValueError("Singular matrix — cannot compute inverse")

            for j in range(2 * n):
                aug[col][j] /= pivot

            for row in range(n):
                if row != col:
                    factor = aug[row][col]
                    for j in range(2 * n):
                        aug[row][j] -= factor * aug[col][j]

        return [[aug[i][j] for j in range(n, 2 * n)] for i in range(n)]


def equilibrium_returns(
    covariance_matrix: List[List[float]],
    market_caps: List[float],
    risk_aversion: float = 2.5
) -> List[float]:
    """
    Compute CAPM equilibrium returns.

    π = δ × Σ × w_mkt
    """
    n = len(covariance_matrix)
    total_cap = sum(market_caps)
    if total_cap <= 0:
        raise ValueError("Total market capitalization must be positive")
    weights = [cap / total_cap for cap in market_caps]

    # Σ × w
    sigma_w = [sum(covariance_matrix[i][j] * weights[j] for j in range(n)) for i in range(n)]

    return [risk_aversion * sw for sw in sigma_w]
