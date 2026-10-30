"""
Changepoint Detection Algorithms

Implements CUSUM (cumulative sum) control charts and Bayesian
online changepoint detection following Adams & MacKay (2007).
All computations are pure Python.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ChangepointResult:
    """Result of changepoint detection.

    Attributes
    ----------
    changepoints : List[int]
        Indices where changepoints were detected.
    scores : List[float]
        Detection score at each time step. Interpretation depends
        on the method (CUSUM statistic or posterior probability).
    method : str
        Name of the detection method used.
    n_changepoints : int
        Number of changepoints detected.
    """

    changepoints: List[int]
    scores: List[float]
    method: str
    n_changepoints: int


# ---------------------------------------------------------------------------
# CUSUM
# ---------------------------------------------------------------------------

class CUSUM:
    """Cumulative Sum (CUSUM) changepoint detector.

    Detects shifts in the mean of a time series using a two-sided
    CUSUM chart. An upward CUSUM tracks positive deviations and a
    downward CUSUM tracks negative deviations from the target mean.

    Parameters
    ----------
    threshold : float
        Decision threshold for declaring a changepoint.
    drift : float
        Allowance parameter (slack). Observations within ``drift``
        of the target mean do not accumulate in the statistic.
    reset : bool
        If True, reset the CUSUM statistic to zero after each
        detected changepoint and continue scanning.

    Examples
    --------
    >>> detector = CUSUM(threshold=3.0, drift=0.5)
    >>> result = detector.detect([0.0] * 50 + [2.0] * 50)
    >>> len(result.changepoints) > 0
    True
    """

    def __init__(
        self,
        threshold: float = 5.0,
        drift: float = 0.0,
        reset: bool = True,
    ) -> None:
        self.threshold = threshold
        self.drift = drift
        self.reset = reset

    def detect(self, series: List[float]) -> ChangepointResult:
        """Detect changepoints in a univariate time series.

        The target mean is estimated from the initial 10% of the
        series (minimum 5 observations). Two-sided CUSUM statistics
        track upward and downward shifts.

        Parameters
        ----------
        series : List[float]
            Univariate time series of length T.

        Returns
        -------
        ChangepointResult
            Detected changepoints and CUSUM scores.

        Raises
        ------
        ValueError
            If the series has fewer than 2 elements.
        """
        n = len(series)
        if n < 2:
            raise ValueError("Series must have at least 2 elements")

        # Estimate target mean from initial segment
        init_len = max(5, n // 10)
        init_len = min(init_len, n)
        mu = sum(series[:init_len]) / init_len

        s_pos = 0.0
        s_neg = 0.0
        changepoints: List[int] = []
        scores: List[float] = [0.0] * n

        for t in range(n):
            s_pos = max(0.0, s_pos + (series[t] - mu - self.drift))
            s_neg = max(0.0, s_neg + (mu - self.drift - series[t]))
            scores[t] = max(s_pos, s_neg)

            if s_pos > self.threshold or s_neg > self.threshold:
                changepoints.append(t)
                if self.reset:
                    s_pos = 0.0
                    s_neg = 0.0

        return ChangepointResult(
            changepoints=changepoints,
            scores=scores,
            method="CUSUM",
            n_changepoints=len(changepoints),
        )


# ---------------------------------------------------------------------------
# Bayesian Online Changepoint Detection
# ---------------------------------------------------------------------------

class BayesianChangepoint:
    """Bayesian online changepoint detection.

    Implements the Adams & MacKay (2007) algorithm with a
    normal-inverse-gamma conjugate prior and Student-t predictive
    distribution.

    Parameters
    ----------
    hazard_rate : float
        Constant hazard rate ``H = 1 / expected_run_length``.
        Default is ``1/250`` (approximately one changepoint per
        trading year).
    prior_mean : float
        Prior mean of the normal distribution (mu_0).
    prior_var : float
        Prior variance scaling parameter (kappa_0).
    threshold : float
        Posterior probability threshold for declaring a changepoint.
        Changepoints are recorded when ``P(run_length=0) > threshold``.

    Examples
    --------
    >>> detector = BayesianChangepoint(hazard_rate=1/50)
    >>> result = detector.detect([0.0]*50 + [3.0]*50)
    >>> len(result.changepoints) > 0
    True
    """

    def __init__(
        self,
        hazard_rate: float = 1.0 / 250.0,
        prior_mean: float = 0.0,
        prior_var: float = 1.0,
        threshold: float = 0.5,
    ) -> None:
        if not 0.0 < hazard_rate < 1.0:
            raise ValueError(
                f"hazard_rate must be in (0, 1), got {hazard_rate}; "
                "use 1 / expected_run_length"
            )
        if prior_var <= 0:
            raise ValueError(f"prior_var must be positive, got {prior_var}")
        self.hazard_rate = hazard_rate
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.threshold = threshold

    def detect(self, series: List[float]) -> ChangepointResult:
        """Detect changepoints using Bayesian online inference.

        Maintains a run length distribution at each time step.
        The predictive distribution is Student-t, derived from a
        normal-inverse-gamma conjugate prior updated online.

        Parameters
        ----------
        series : List[float]
            Univariate time series of length T.

        Returns
        -------
        ChangepointResult
            Detected changepoints and posterior changepoint
            probabilities at each time step.

        Raises
        ------
        ValueError
            If the series has fewer than 2 elements.
        """
        n = len(series)
        if n < 2:
            raise ValueError("Series must have at least 2 elements")

        H = self.hazard_rate

        # Sufficient statistics for normal-inverse-gamma prior
        # For each possible run length r we track:
        #   mu_r, kappa_r, alpha_r, beta_r
        # These parameterise the posterior after observing r
        # data points within the current segment.
        mu0 = self.prior_mean
        kappa0 = 1.0 / max(self.prior_var, 1e-300)
        alpha0 = 1.0
        beta0 = self.prior_var * 0.5

        # Arrays indexed by run length (grow each step)
        mu_params: List[float] = [mu0]
        kappa_params: List[float] = [kappa0]
        alpha_params: List[float] = [alpha0]
        beta_params: List[float] = [beta0]

        # Run length distribution: R[r] = P(r_t = r)
        # At t=0, P(r_0 = 0) = 1
        R: List[float] = [1.0]

        changepoints: List[int] = []
        scores: List[float] = [0.0] * n
        prev_map_rl = 0
        candidate: Optional[int] = None

        for t in range(n):
            x = series[t]
            curr_len = len(R)  # max run length + 1

            # Step 1: Evaluate predictive probabilities for each
            # run length using Student-t distribution
            pi: List[float] = []
            for r in range(curr_len):
                pi.append(
                    _student_t_pdf(
                        x,
                        mu_params[r],
                        beta_params[r]
                        * (kappa_params[r] + 1.0)
                        / (alpha_params[r] * kappa_params[r]),
                        2.0 * alpha_params[r],
                    )
                )

            # Step 2: Growth probabilities (run continues)
            # R_new[r+1] = R_old[r] * (1 - H) * pi[r]
            new_R: List[float] = [0.0] * (curr_len + 1)
            for r in range(curr_len):
                new_R[r + 1] = R[r] * (1.0 - H) * pi[r]

            # Step 3: Changepoint probability (run resets to 0)
            # R_new[0] = sum_r R_old[r] * H * pi[r]
            cp_prob = 0.0
            for r in range(curr_len):
                cp_prob += R[r] * H * pi[r]
            new_R[0] = cp_prob

            # Step 4: Normalise run length distribution
            evidence = sum(new_R)
            if evidence > 0:
                new_R = [v / evidence for v in new_R]

            R = new_R
            # With a constant hazard, P(r_t = 0) normalises to exactly H and
            # carries no information. A changepoint at t shows up as posterior
            # mass on the run that starts at observation t, i.e. r_t <= 1.
            scores[t] = R[0] + R[1]

            # Read changepoints off the MAP run length: when it resets, the
            # new run started at t - map_rl + 1. The candidate is confirmed once
            # the posterior mass on runs no longer than the MAP exceeds the
            # threshold, and dropped if the MAP returns to an older run.
            map_rl = max(range(len(R)), key=R.__getitem__)
            run_start = t - map_rl + 1
            if t > 0 and map_rl < prev_map_rl:
                candidate = run_start
            if candidate is not None:
                if run_start != candidate:
                    candidate = None
                elif sum(R[: map_rl + 1]) > self.threshold:
                    if candidate > 0 and (not changepoints or candidate > changepoints[-1]):
                        changepoints.append(candidate)
                    candidate = None
            prev_map_rl = map_rl

            # Step 5: Update sufficient statistics
            # Add new prior entry for run length 0 (reset)
            new_mu: List[float] = [mu0]
            new_kappa: List[float] = [kappa0]
            new_alpha: List[float] = [alpha0]
            new_beta: List[float] = [beta0]

            # Update existing entries (run length grows by 1)
            for r in range(curr_len):
                kappa_new = kappa_params[r] + 1.0
                mu_new = (
                    kappa_params[r] * mu_params[r] + x
                ) / kappa_new
                alpha_new = alpha_params[r] + 0.5
                beta_new = (
                    beta_params[r]
                    + 0.5
                    * kappa_params[r]
                    * (x - mu_params[r]) ** 2
                    / kappa_new
                )
                new_mu.append(mu_new)
                new_kappa.append(kappa_new)
                new_alpha.append(alpha_new)
                new_beta.append(beta_new)

            mu_params = new_mu
            kappa_params = new_kappa
            alpha_params = new_alpha
            beta_params = new_beta

            # Pruning: remove negligibly small run lengths to
            # keep memory bounded. Keep at most max(500, 2*t) entries.
            max_rl = max(500, 2 * (t + 1))
            if len(R) > max_rl:
                R = R[:max_rl]
                mu_params = mu_params[:max_rl]
                kappa_params = kappa_params[:max_rl]
                alpha_params = alpha_params[:max_rl]
                beta_params = beta_params[:max_rl]
                # Renormalise
                s = sum(R)
                if s > 0:
                    R = [v / s for v in R]

        return ChangepointResult(
            changepoints=changepoints,
            scores=scores,
            method="BayesianChangepoint",
            n_changepoints=len(changepoints),
        )


# ---------------------------------------------------------------------------
# Helper: Student-t PDF
# ---------------------------------------------------------------------------

def _student_t_pdf(
    x: float, mu: float, var: float, nu: float
) -> float:
    """Evaluate the Student-t probability density function.

    Parameters
    ----------
    x : float
        Observation.
    mu : float
        Location parameter.
    var : float
        Scale parameter (variance-like). Must be > 0.
    nu : float
        Degrees of freedom. Must be > 0.

    Returns
    -------
    float
        Density p(x | mu, var, nu).
    """
    var = max(var, 1e-300)
    nu = max(nu, 1e-300)
    sigma = math.sqrt(var)

    # log Gamma((nu+1)/2) - log Gamma(nu/2)
    log_coeff = (
        math.lgamma((nu + 1.0) / 2.0)
        - math.lgamma(nu / 2.0)
        - 0.5 * math.log(nu * math.pi * var)
    )
    z = (x - mu) / sigma
    log_body = -(nu + 1.0) / 2.0 * math.log(1.0 + z * z / nu)

    return math.exp(log_coeff + log_body)
