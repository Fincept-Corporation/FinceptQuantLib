"""
Gaussian Process Volatility Surface

Provides a 2D Gaussian process regression model for fitting
volatility surfaces over (strike, expiry) space. Inherits from
VolatilitySurface ABC to provide the standard interface.

Features:
- Anisotropic 2D RBF kernel with separate length scales for
  strike and expiry dimensions
- Posterior uncertainty quantification at any point
- Optional hyperparameter optimization via grid search over
  log marginal likelihood
- Cholesky-based numerically stable GP inference
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from finceptquantlib.volatility.surface import VolatilitySurface


@dataclass
class GPVolSurfaceResult:
    """Result of GP volatility surface fitting.

    Attributes
    ----------
    log_marginal_likelihood : float
        Log marginal likelihood of the fitted model.
    kernel_params : Dict[str, float]
        Optimized kernel hyperparameters.
    n_points_fitted : int
        Number of observation points used for fitting.
    """

    log_marginal_likelihood: float
    kernel_params: Dict[str, float]
    n_points_fitted: int


def _cholesky(A: List[List[float]]) -> List[List[float]]:
    """Compute lower-triangular Cholesky factor L where A = L L^T.

    Parameters
    ----------
    A : List[List[float]]
        Symmetric positive-definite matrix.

    Returns
    -------
    List[List[float]]
        Lower-triangular Cholesky factor.

    Raises
    ------
    ValueError
        If matrix is not positive definite.
    """
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                if val <= 0.0:
                    raise ValueError(
                        f"Matrix not positive definite at index {i}."
                    )
                L[i][j] = math.sqrt(val)
            else:
                if L[j][j] == 0.0:
                    raise ValueError(
                        "Zero diagonal in Cholesky factor."
                    )
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L


def _forward_solve(
    L: List[List[float]], b: List[float]
) -> List[float]:
    """Solve L x = b where L is lower triangular.

    Parameters
    ----------
    L : List[List[float]]
        Lower-triangular matrix.
    b : List[float]
        Right-hand side vector.

    Returns
    -------
    List[float]
        Solution vector.
    """
    n = len(b)
    x = [0.0] * n
    for i in range(n):
        s = sum(L[i][k] * x[k] for k in range(i))
        x[i] = (b[i] - s) / L[i][i]
    return x


def _backward_solve(
    L: List[List[float]], b: List[float]
) -> List[float]:
    """Solve L^T x = b where L is lower triangular.

    Parameters
    ----------
    L : List[List[float]]
        Lower-triangular matrix.
    b : List[float]
        Right-hand side vector.

    Returns
    -------
    List[float]
        Solution vector.
    """
    n = len(b)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(L[k][i] * x[k] for k in range(i + 1, n))
        x[i] = (b[i] - s) / L[i][i]
    return x


def _cholesky_solve(
    L: List[List[float]], b: List[float]
) -> List[float]:
    """Solve A x = b given Cholesky factor L of A.

    Parameters
    ----------
    L : List[List[float]]
        Lower-triangular Cholesky factor.
    b : List[float]
        Right-hand side vector.

    Returns
    -------
    List[float]
        Solution A^{-1} b.
    """
    y = _forward_solve(L, b)
    return _backward_solve(L, y)


def _log_det_from_cholesky(L: List[List[float]]) -> float:
    """Log determinant from Cholesky factor.

    Parameters
    ----------
    L : List[List[float]]
        Lower-triangular Cholesky factor.

    Returns
    -------
    float
        log|A| = 2 * sum(log(L_ii)).
    """
    return 2.0 * sum(math.log(L[i][i]) for i in range(len(L)))


class GPVolSurface(VolatilitySurface):
    """2D Gaussian process volatility surface.

    Fits a GP regression model to scattered (strike, expiry, vol)
    observations using an anisotropic 2D RBF kernel. Inherits from
    VolatilitySurface to provide the standard interface.

    Parameters
    ----------
    kernel : str
        Kernel type. Currently supports 'rbf_2d'.
    length_scale_strike : float
        RBF length scale for the strike dimension.
    length_scale_expiry : float
        RBF length scale for the expiry dimension.
    signal_variance : float
        Kernel signal variance (output scale).
    noise_variance : float
        Observation noise variance.
    optimize_hyperparams : bool
        Whether to optimize hyperparameters via grid search
        on the log marginal likelihood.

    Examples
    --------
    >>> gp = GPVolSurface()
    >>> strikes = [0.8, 0.9, 1.0, 1.1, 1.2]
    >>> expiries = [0.25, 0.5, 1.0, 2.0, 5.0]
    >>> vols = [0.25, 0.22, 0.20, 0.22, 0.25]
    >>> result = gp.fit(strikes, expiries, vols)
    >>> gp.volatility(1.0, 0.5)
    0.216146
    """

    def __init__(
        self,
        kernel: str = "rbf_2d",
        length_scale_strike: float = 0.1,
        length_scale_expiry: float = 1.0,
        signal_variance: float = 1.0,
        noise_variance: float = 1e-4,
        optimize_hyperparams: bool = True,
    ) -> None:
        if kernel.lower() != "rbf_2d":
            raise ValueError(
                f"Unknown kernel '{kernel}'. Supported: 'rbf_2d'."
            )
        self._kernel = kernel
        self._ls_strike = length_scale_strike
        self._ls_expiry = length_scale_expiry
        self._signal_variance = signal_variance
        self._noise_variance = noise_variance
        self._optimize = optimize_hyperparams

        self._fitted = False
        self._train_strikes: List[float] = []
        self._train_expiries: List[float] = []
        self._train_vols: List[float] = []
        self._alpha: List[float] = []
        self._L: List[List[float]] = []

    def _kernel_func(
        self,
        s1: float,
        e1: float,
        s2: float,
        e2: float,
        ls_s: float,
        ls_e: float,
        sig_var: float,
    ) -> float:
        """Evaluate 2D anisotropic RBF kernel.

        k((s1,e1), (s2,e2)) = sigma^2 * exp(-0.5 *
            ((s1-s2)^2/l_s^2 + (e1-e2)^2/l_e^2))

        Parameters
        ----------
        s1, e1 : float
            First point (strike, expiry).
        s2, e2 : float
            Second point (strike, expiry).
        ls_s : float
            Length scale for strike.
        ls_e : float
            Length scale for expiry.
        sig_var : float
            Signal variance.

        Returns
        -------
        float
            Kernel value.
        """
        ds = (s1 - s2) / ls_s
        de = (e1 - e2) / ls_e
        return sig_var * math.exp(-0.5 * (ds * ds + de * de))

    def _build_kernel_matrix(
        self,
        S1: List[float],
        E1: List[float],
        S2: List[float],
        E2: List[float],
        ls_s: float,
        ls_e: float,
        sig_var: float,
    ) -> List[List[float]]:
        """Build kernel matrix between two sets of points.

        Parameters
        ----------
        S1, E1 : List[float]
            First set (strikes, expiries).
        S2, E2 : List[float]
            Second set (strikes, expiries).
        ls_s : float
            Strike length scale.
        ls_e : float
            Expiry length scale.
        sig_var : float
            Signal variance.

        Returns
        -------
        List[List[float]]
            Kernel matrix (n1 x n2).
        """
        n1 = len(S1)
        n2 = len(S2)
        K = [[0.0] * n2 for _ in range(n1)]
        for i in range(n1):
            for j in range(n2):
                K[i][j] = self._kernel_func(
                    S1[i], E1[i], S2[j], E2[j],
                    ls_s, ls_e, sig_var,
                )
        return K

    def _compute_lml(
        self,
        strikes: List[float],
        expiries: List[float],
        vols: List[float],
        ls_s: float,
        ls_e: float,
        sig_var: float,
        noise_var: float,
    ) -> float:
        """Compute log marginal likelihood.

        Parameters
        ----------
        strikes, expiries, vols : List[float]
            Training data.
        ls_s, ls_e : float
            Length scales.
        sig_var : float
            Signal variance.
        noise_var : float
            Noise variance.

        Returns
        -------
        float
            Log marginal likelihood.
        """
        n = len(strikes)
        K = self._build_kernel_matrix(
            strikes, expiries, strikes, expiries,
            ls_s, ls_e, sig_var,
        )
        for i in range(n):
            K[i][i] += noise_var

        try:
            L = _cholesky(K)
        except ValueError:
            return float("-inf")

        alpha = _cholesky_solve(L, vols)
        data_fit = -0.5 * sum(vols[i] * alpha[i] for i in range(n))
        complexity = -0.5 * _log_det_from_cholesky(L)
        constant = -0.5 * n * math.log(2.0 * math.pi)
        return data_fit + complexity + constant

    def _optimize_hyperparams(
        self,
        strikes: List[float],
        expiries: List[float],
        vols: List[float],
    ) -> None:
        """Optimize hyperparameters via grid search.

        Parameters
        ----------
        strikes, expiries, vols : List[float]
            Training data.
        """
        s_range = (
            max(strikes) - min(strikes) if len(strikes) > 1 else 1.0
        )
        e_range = (
            max(expiries) - min(expiries)
            if len(expiries) > 1
            else 1.0
        )
        vol_var = (
            sum(
                (v - sum(vols) / len(vols)) ** 2 for v in vols
            ) / len(vols)
        ) if vols else 1.0

        ls_s_cands = [s_range * f for f in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]]
        ls_e_cands = [e_range * f for f in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]]
        sv_cands = [vol_var * f for f in [0.5, 1.0, 2.0, 5.0, 10.0]]
        nv_cands = [1e-6, 1e-5, 1e-4, 1e-3]

        best_lml = float("-inf")
        best_ls_s = self._ls_strike
        best_ls_e = self._ls_expiry
        best_sv = self._signal_variance
        best_nv = self._noise_variance

        for ls_s in ls_s_cands:
            for ls_e in ls_e_cands:
                for sv in sv_cands:
                    for nv in nv_cands:
                        lml = self._compute_lml(
                            strikes, expiries, vols,
                            ls_s, ls_e, sv, nv,
                        )
                        if lml > best_lml:
                            best_lml = lml
                            best_ls_s = ls_s
                            best_ls_e = ls_e
                            best_sv = sv
                            best_nv = nv

        self._ls_strike = best_ls_s
        self._ls_expiry = best_ls_e
        self._signal_variance = best_sv
        self._noise_variance = best_nv

    def fit(
        self,
        strikes: List[float],
        expiries: List[float],
        vols: List[float],
    ) -> GPVolSurfaceResult:
        """Fit the GP model to observed volatilities.

        Parameters
        ----------
        strikes : List[float]
            Strike values for each observation.
        expiries : List[float]
            Expiry values for each observation.
        vols : List[float]
            Observed volatilities.

        Returns
        -------
        GPVolSurfaceResult
            Fitting result with hyperparameters and likelihood.

        Raises
        ------
        ValueError
            If inputs have inconsistent lengths or are empty.
        """
        if not (len(strikes) == len(expiries) == len(vols)):
            raise ValueError(
                "strikes, expiries, and vols must have the same "
                "length."
            )
        if len(strikes) == 0:
            raise ValueError("Cannot fit with empty data.")

        self._train_strikes = list(strikes)
        self._train_expiries = list(expiries)
        self._train_vols = list(vols)
        n = len(strikes)

        if self._optimize:
            self._optimize_hyperparams(strikes, expiries, vols)

        K = self._build_kernel_matrix(
            strikes, expiries, strikes, expiries,
            self._ls_strike, self._ls_expiry,
            self._signal_variance,
        )
        for i in range(n):
            K[i][i] += self._noise_variance

        self._L = _cholesky(K)
        self._alpha = _cholesky_solve(self._L, vols)

        lml = self._compute_lml(
            strikes, expiries, vols,
            self._ls_strike, self._ls_expiry,
            self._signal_variance, self._noise_variance,
        )

        self._fitted = True

        return GPVolSurfaceResult(
            log_marginal_likelihood=lml,
            kernel_params={
                "length_scale_strike": self._ls_strike,
                "length_scale_expiry": self._ls_expiry,
                "signal_variance": self._signal_variance,
                "noise_variance": self._noise_variance,
            },
            n_points_fitted=n,
        )

    def _check_fitted(self) -> None:
        """Verify model has been fitted.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError(
                "Model has not been fitted. Call fit() first."
            )

    def _predict_mean(
        self, strike: float, expiry: float
    ) -> float:
        """Compute GP posterior mean at a query point.

        mu(x*) = k(x*, X)^T alpha

        Parameters
        ----------
        strike : float
            Query strike.
        expiry : float
            Query expiry.

        Returns
        -------
        float
            Posterior mean volatility.
        """
        k_star = [
            self._kernel_func(
                strike, expiry,
                self._train_strikes[i], self._train_expiries[i],
                self._ls_strike, self._ls_expiry,
                self._signal_variance,
            )
            for i in range(len(self._train_strikes))
        ]
        return sum(
            k_star[i] * self._alpha[i]
            for i in range(len(self._alpha))
        )

    def volatility(
        self,
        strike: float,
        expiry: float,
        forward: Optional[float] = None,
    ) -> float:
        """Get volatility for given strike and expiry.

        Parameters
        ----------
        strike : float
            Option strike.
        expiry : float
            Option expiry in years.
        forward : float, optional
            Forward price (unused in direct GP model).

        Returns
        -------
        float
            Predicted volatility.
        """
        self._check_fitted()
        return self._predict_mean(strike, expiry)

    def total_variance(
        self,
        strike: float,
        expiry: float,
        forward: Optional[float] = None,
    ) -> float:
        """Get total variance (vol^2 * T).

        Parameters
        ----------
        strike : float
            Option strike.
        expiry : float
            Option expiry in years.
        forward : float, optional
            Forward price (unused).

        Returns
        -------
        float
            Total variance.
        """
        vol = self.volatility(strike, expiry, forward)
        return vol * vol * expiry

    def uncertainty(
        self, strike: float, expiry: float
    ) -> float:
        """Get posterior standard deviation at a query point.

        var(x*) = k(x*,x*) - k(x*,X)^T K_y^{-1} k(X,x*)

        Parameters
        ----------
        strike : float
            Query strike.
        expiry : float
            Query expiry.

        Returns
        -------
        float
            Posterior standard deviation.
        """
        self._check_fitted()

        k_star = [
            self._kernel_func(
                strike, expiry,
                self._train_strikes[i], self._train_expiries[i],
                self._ls_strike, self._ls_expiry,
                self._signal_variance,
            )
            for i in range(len(self._train_strikes))
        ]
        k_ss = self._kernel_func(
            strike, expiry, strike, expiry,
            self._ls_strike, self._ls_expiry,
            self._signal_variance,
        )

        v = _forward_solve(self._L, k_star)
        var = k_ss - sum(vi * vi for vi in v)
        return math.sqrt(max(var, 0.0))
