"""
Credit Rating Migration Analysis

Implements transition matrix estimation, multi-period projection,
generator matrix computation, and cumulative default probability
extraction for credit rating migration analysis.

Used in regulatory capital models (Basel IRB approach) and
portfolio credit risk assessment.

All computations are pure Python with no external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class MigrationMatrixResult:
    """Result of migration matrix estimation.

    Attributes
    ----------
    matrix : List[List[float]]
        Transition probability matrix (n_ratings x n_ratings).
        Row i, column j = P(migrate from rating i to rating j).
        Rows sum to 1.
    ratings : List[str]
        Rating labels in order (e.g., ["AAA", ..., "D"]).
    n_observations : int
        Total number of rating transitions observed.
    generator : Optional[List[List[float]]]
        Generator (intensity) matrix if computed.
    """

    matrix: List[List[float]]
    ratings: List[str]
    n_observations: int
    generator: Optional[List[List[float]]] = None


class RatingMigration:
    """Credit rating migration analysis.

    Estimates transition matrices from observed rating migrations,
    computes multi-period transition probabilities, derives
    generator matrices, and extracts default probabilities.

    Parameters
    ----------
    ratings : List[str]
        Ordered list of rating categories from best to worst.
        The last rating is assumed to be the default/absorbing
        state (e.g., ``"D"``).

    Examples
    --------
    >>> rm = RatingMigration(
    ...     ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]
    ... )
    >>> result = rm.estimate(start_ratings, end_ratings)  # doctest: +SKIP
    >>> multi = rm.multi_period(result.matrix, 5)  # doctest: +SKIP
    """

    def __init__(self, ratings: List[str]) -> None:
        if len(ratings) < 2:
            raise ValueError(
                "At least 2 rating categories required"
            )
        self.ratings = ratings
        self._rating_index: Dict[str, int] = {
            r: i for i, r in enumerate(ratings)
        }
        self._n = len(ratings)

    def estimate(
        self,
        start_ratings: List[str],
        end_ratings: List[str],
    ) -> MigrationMatrixResult:
        """Estimate transition matrix from observed migrations.

        Counts transitions between each pair of rating states
        and normalizes each row to sum to 1 (cohort method).

        Parameters
        ----------
        start_ratings : List[str]
            Beginning-of-period ratings for each obligor.
        end_ratings : List[str]
            End-of-period ratings for each obligor.

        Returns
        -------
        MigrationMatrixResult
            Estimated transition matrix.

        Raises
        ------
        ValueError
            If inputs are invalid or contain unknown ratings.
        """
        if len(start_ratings) != len(end_ratings):
            raise ValueError(
                "start_ratings and end_ratings must have "
                "same length"
            )
        if len(start_ratings) == 0:
            raise ValueError("Inputs must not be empty")

        n = self._n
        counts = [[0] * n for _ in range(n)]

        for s, e in zip(start_ratings, end_ratings):
            if s not in self._rating_index:
                raise ValueError(f"Unknown rating: '{s}'")
            if e not in self._rating_index:
                raise ValueError(f"Unknown rating: '{e}'")
            i = self._rating_index[s]
            j = self._rating_index[e]
            counts[i][j] += 1

        return self.matrix_from_counts(counts)

    def matrix_from_counts(
        self, counts: List[List[float]]
    ) -> MigrationMatrixResult:
        """Estimate the transition matrix from a cohort count matrix.

        Each row is normalized to sum to 1 (cohort method); a rating
        with no observations keeps an identity row.

        Parameters
        ----------
        counts : List[List[float]]
            counts[i][j] = number of obligors that moved from
            rating i to rating j, in the order of ``ratings``.

        Returns
        -------
        MigrationMatrixResult
            Estimated transition matrix.

        Raises
        ------
        ValueError
            If the matrix is not n_ratings x n_ratings or has
            negative counts.
        """
        n = self._n
        if len(counts) != n or any(len(row) != n for row in counts):
            raise ValueError(
                f"counts must be a {n}x{n} matrix (one row and "
                f"column per rating)"
            )
        if any(c < 0 for row in counts for c in row):
            raise ValueError("counts must be non-negative")

        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            row_sum = sum(counts[i])
            if row_sum > 0:
                for j in range(n):
                    matrix[i][j] = counts[i][j] / row_sum
            else:
                # No observations: identity row
                matrix[i][i] = 1.0

        return MigrationMatrixResult(
            matrix=matrix,
            ratings=self.ratings[:],
            n_observations=int(sum(sum(row) for row in counts)),
        )

    def multi_period(
        self,
        matrix: List[List[float]],
        periods: int,
    ) -> List[List[float]]:
        """Compute multi-period transition matrix via matrix power.

        M^n computed by repeated matrix multiplication.

        Parameters
        ----------
        matrix : List[List[float]]
            One-period transition matrix.
        periods : int
            Number of periods (must be >= 1).

        Returns
        -------
        List[List[float]]
            Multi-period transition matrix M^periods.

        Raises
        ------
        ValueError
            If periods < 1.
        """
        if periods < 1:
            raise ValueError("periods must be >= 1")

        n = len(matrix)

        # Start with identity
        result = [
            [1.0 if i == j else 0.0 for j in range(n)]
            for i in range(n)
        ]

        # Repeated squaring for efficiency
        base = [row[:] for row in matrix]
        p = periods

        while p > 0:
            if p % 2 == 1:
                result = self._mat_mul(result, base)
            base = self._mat_mul(base, base)
            p //= 2

        return result

    def generator_matrix(
        self, matrix: List[List[float]]
    ) -> List[List[float]]:
        """Compute generator (intensity) matrix from transition matrix.

        Uses the matrix logarithm via series expansion:
            Q = sum_{k=1}^{N} (-1)^{k+1} * (P - I)^k / k

        Then adjusts to ensure:
        - Off-diagonal elements >= 0
        - Row sums equal 0

        The generator Q satisfies P = exp(Q) and represents
        continuous-time transition intensities.

        Parameters
        ----------
        matrix : List[List[float]]
            Transition probability matrix.

        Returns
        -------
        List[List[float]]
            Generator matrix Q.
        """
        n = len(matrix)

        # Compute P - I
        p_minus_i = [
            [
                matrix[i][j] - (1.0 if i == j else 0.0)
                for j in range(n)
            ]
            for i in range(n)
        ]

        # Series expansion: Q = sum_{k=1}^{N} (-1)^{k+1} (P-I)^k / k
        # Use N = 50 terms for convergence
        max_terms = 50
        Q = [[0.0] * n for _ in range(n)]

        # power_k = (P - I)^k, starts at k=1
        power_k = [row[:] for row in p_minus_i]

        for k in range(1, max_terms + 1):
            sign = 1.0 if k % 2 == 1 else -1.0
            coeff = sign / k

            for i in range(n):
                for j in range(n):
                    Q[i][j] += coeff * power_k[i][j]

            # Update power: (P-I)^{k+1} = (P-I)^k * (P-I)
            if k < max_terms:
                power_k = self._mat_mul(power_k, p_minus_i)

        # Ensure off-diagonal >= 0 and rows sum to 0
        for i in range(n):
            for j in range(n):
                if i != j and Q[i][j] < 0.0:
                    Q[i][j] = 0.0

        # Adjust diagonal so rows sum to 0
        for i in range(n):
            off_diag_sum = sum(
                Q[i][j] for j in range(n) if j != i
            )
            Q[i][i] = -off_diag_sum

        return Q

    def default_probability(
        self,
        matrix: List[List[float]],
        rating_idx: int,
        horizon: int,
    ) -> float:
        """Extract cumulative default probability.

        Computes the probability that an obligor currently at
        ``rating_idx`` will be in the default state (last column)
        within the given horizon.

        Parameters
        ----------
        matrix : List[List[float]]
            One-period transition matrix.
        rating_idx : int
            Index of the current rating (0-based).
        horizon : int
            Number of periods for cumulative PD.

        Returns
        -------
        float
            Cumulative probability of default.

        Raises
        ------
        ValueError
            If rating_idx is out of range or horizon < 1.
        """
        n = len(matrix)
        if rating_idx < 0 or rating_idx >= n:
            raise ValueError(
                f"rating_idx must be in [0, {n - 1}]"
            )
        if horizon < 1:
            raise ValueError("horizon must be >= 1")

        multi = self.multi_period(matrix, horizon)
        # Default is last column
        return multi[rating_idx][n - 1]

    @staticmethod
    def _mat_mul(
        A: List[List[float]], B: List[List[float]]
    ) -> List[List[float]]:
        """Multiply two square matrices.

        Parameters
        ----------
        A : List[List[float]]
            First matrix (n x n).
        B : List[List[float]]
            Second matrix (n x n).

        Returns
        -------
        List[List[float]]
            Product matrix (n x n).
        """
        n = len(A)
        result = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s = 0.0
                for k in range(n):
                    s += A[i][k] * B[k][j]
                result[i][j] = s
        return result
