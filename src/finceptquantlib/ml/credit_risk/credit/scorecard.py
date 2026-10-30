"""
Credit Scorecard Construction

Converts a logistic regression model with WoE-transformed features
into a points-based credit scorecard. Implements the standard
scorecard scaling methodology:

    Score = Offset - Factor * ln(odds)

where:
    Factor = PDO / ln(2)
    Offset = base_score - Factor * ln(base_odds)

Each feature bin receives points proportional to its contribution
to the log-odds via the logistic regression coefficients and WoE
values.

All computations are pure Python with no external dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from finceptquantlib.ml.credit_risk.credit.logistic import LogisticRegression
from finceptquantlib.ml.credit_risk.credit.woe import WoEBinner


@dataclass
class ScorecardResult:
    """Result of scorecard construction.

    Attributes
    ----------
    base_score : float
        Base score for the scorecard.
    pdo : float
        Points to Double the Odds.
    base_odds : float
        Base odds (non-event:event ratio) at the base score.
    feature_points : Dict[str, List[Tuple[str, float]]]
        For each feature name, a list of (bin_description, points)
        tuples.
    total_iv : float
        Sum of Information Values across all features.
    min_score : float
        Theoretical minimum score.
    max_score : float
        Theoretical maximum score.
    """

    base_score: float
    pdo: float
    base_odds: float
    feature_points: Dict[str, List[Tuple[str, float]]]
    total_iv: float
    min_score: float
    max_score: float


class Scorecard:
    """Credit scorecard builder.

    Constructs a points-based credit scorecard from a fitted
    logistic regression model and WoE binners. Uses the standard
    PDO (Points to Double the Odds) scaling methodology common
    in consumer credit risk.

    Parameters
    ----------
    pdo : float
        Points to Double the Odds. A typical value is 50.
        Higher PDO means a wider score range.
    base_score : float
        Base score corresponding to the base odds. Typical
        value is 600.
    base_odds : float
        Base odds (good:bad ratio) at the base score.
        For example, 50 means 50 non-events per 1 event.

    Examples
    --------
    >>> sc = Scorecard(pdo=50.0, base_score=600.0, base_odds=50.0)
    >>> result = sc.build(model, woe_binners, feature_names)  # doctest: +SKIP
    >>> score = sc.score({"income": 50000, "age": 35})  # doctest: +SKIP
    """

    def __init__(
        self,
        pdo: float = 50.0,
        base_score: float = 600.0,
        base_odds: float = 50.0,
    ) -> None:
        if pdo <= 0:
            raise ValueError("pdo must be positive")
        if base_odds <= 0:
            raise ValueError("base_odds must be positive")

        self.pdo = pdo
        self.base_score = base_score
        self.base_odds = base_odds

        # Derived constants
        self.factor = pdo / math.log(2.0)
        self.offset = base_score - self.factor * math.log(
            base_odds
        )

        # Built scorecard state
        self._result: Optional[ScorecardResult] = None
        self._model: Optional[LogisticRegression] = None
        self._binners: Optional[Dict[str, WoEBinner]] = None
        self._feature_names: Optional[List[str]] = None

    def build(
        self,
        model: LogisticRegression,
        woe_binners: Dict[str, WoEBinner],
        feature_names: List[str],
    ) -> ScorecardResult:
        """Build the scorecard from model and WoE binners.

        Computes points for each bin of each feature using:
            Points_ij = -(WoE_ij * coef_i + intercept / n) * Factor

        where i is the feature index, j is the bin index, n is
        the number of features, and Factor = PDO / ln(2).

        Parameters
        ----------
        model : LogisticRegression
            Fitted logistic regression model.
        woe_binners : Dict[str, WoEBinner]
            Fitted WoE binners keyed by feature name.
        feature_names : List[str]
            Feature names in the order used during model fitting.

        Returns
        -------
        ScorecardResult
            Complete scorecard with points for each bin.

        Raises
        ------
        RuntimeError
            If model or any binner has not been fitted.
        ValueError
            If feature names don't match model coefficients.
        """
        if model._fit_result is None:
            raise RuntimeError("Model has not been fitted")

        coefficients = model._fit_result.coefficients
        intercept = model._fit_result.intercept

        if len(feature_names) != len(coefficients):
            raise ValueError(
                f"feature_names length ({len(feature_names)}) "
                f"must match number of coefficients "
                f"({len(coefficients)})"
            )

        n_features = len(feature_names)
        intercept_share = intercept / n_features

        feature_points: Dict[str, List[Tuple[str, float]]] = {}
        total_iv = 0.0
        min_score = self.offset
        max_score = self.offset

        for i, name in enumerate(feature_names):
            if name not in woe_binners:
                raise ValueError(
                    f"WoE binner not found for feature '{name}'"
                )
            binner = woe_binners[name]
            if binner._bin_result is None:
                raise RuntimeError(
                    f"WoEBinner for '{name}' has not been fitted"
                )

            bin_result = binner._bin_result
            coef = coefficients[i]
            total_iv += bin_result.total_iv

            bin_points: List[Tuple[str, float]] = []
            min_pts = float("inf")
            max_pts = float("-inf")

            for j in range(len(bin_result.woe_values)):
                woe = bin_result.woe_values[j]

                # Points for this bin
                points = -(
                    (woe * coef + intercept_share) * self.factor
                )

                # Bin description
                if j == 0:
                    desc = f"<= {bin_result.boundaries[1]:.4f}"
                elif j == len(bin_result.woe_values) - 1:
                    desc = f"> {bin_result.boundaries[j]:.4f}"
                else:
                    desc = (
                        f"({bin_result.boundaries[j]:.4f}, "
                        f"{bin_result.boundaries[j + 1]:.4f}]"
                    )

                bin_points.append((desc, points))
                min_pts = min(min_pts, points)
                max_pts = max(max_pts, points)

            feature_points[name] = bin_points
            min_score += min_pts
            max_score += max_pts

        self._result = ScorecardResult(
            base_score=self.base_score,
            pdo=self.pdo,
            base_odds=self.base_odds,
            feature_points=feature_points,
            total_iv=total_iv,
            min_score=min_score,
            max_score=max_score,
        )
        self._model = model
        self._binners = woe_binners
        self._feature_names = feature_names

        return self._result

    def score(self, applicant: Dict[str, float]) -> float:
        """Score an applicant using the built scorecard.

        Looks up the appropriate bin for each feature value
        and sums the corresponding points, adding the offset.

        Parameters
        ----------
        applicant : Dict[str, float]
            Feature values keyed by feature name.

        Returns
        -------
        float
            Total credit score.

        Raises
        ------
        RuntimeError
            If scorecard has not been built.
        ValueError
            If required features are missing.
        """
        if self._result is None or self._binners is None:
            raise RuntimeError("Scorecard has not been built")
        if self._feature_names is None:
            raise RuntimeError("Scorecard has not been built")

        total_score = self.offset

        for i, name in enumerate(self._feature_names):
            if name not in applicant:
                raise ValueError(
                    f"Missing feature '{name}' in applicant"
                )

            value = applicant[name]
            binner = self._binners[name]
            bin_result = binner._bin_result
            if bin_result is None:
                raise RuntimeError(
                    f"WoEBinner for '{name}' not fitted"
                )

            # Find which bin the value falls into
            boundaries = bin_result.boundaries
            n_bins = len(boundaries) - 1
            bin_idx = n_bins - 1  # default last bin
            for j in range(n_bins):
                if j == n_bins - 1:
                    bin_idx = j
                    break
                if value <= boundaries[j + 1]:
                    bin_idx = j
                    break

            # Look up points
            points = self._result.feature_points[name][bin_idx][1]
            total_score += points

        return total_score

    def score_to_pd(self, score: float) -> float:
        """Convert credit score to probability of default.

        PD = 1 / (1 + exp((score - offset) / factor))

        Parameters
        ----------
        score : float
            Credit score.

        Returns
        -------
        float
            Probability of default in (0, 1).
        """
        z = (score - self.offset) / self.factor
        z = max(-500.0, min(500.0, z))
        return 1.0 / (1.0 + math.exp(z))

    def pd_to_score(self, pd: float) -> float:
        """Convert probability of default to credit score.

        Score = offset + factor * ln((1 - pd) / pd)

        Parameters
        ----------
        pd : float
            Probability of default in (0, 1).

        Returns
        -------
        float
            Credit score.

        Raises
        ------
        ValueError
            If pd is not in (0, 1).
        """
        if pd <= 0.0 or pd >= 1.0:
            raise ValueError("pd must be in (0, 1)")
        odds = (1.0 - pd) / pd
        return self.offset + self.factor * math.log(odds)
