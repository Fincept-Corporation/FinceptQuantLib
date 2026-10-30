"""
Game Theory

Strategic interaction models:
- Normal form games
- Nash equilibrium computation
- Mixed strategy equilibria
- Dominated strategy elimination
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Strategy:
    """A pure strategy in a game."""
    name: str
    index: int


@dataclass
class Player:
    """A player in a game."""
    name: str
    strategies: List[Strategy]

    def __post_init__(self) -> None:
        # Assign indices to strategies
        for i, s in enumerate(self.strategies):
            s.index = i


@dataclass
class NormalFormGame:
    """
    Normal (strategic) form game.

    Represents a simultaneous-move game with payoff matrices.

    Attributes:
        players: List of players
        payoff_matrices: Dict mapping player index to payoff matrix
    """
    players: List[Player]
    payoff_matrices: Dict[int, List[List[float]]]

    def __post_init__(self) -> None:
        self.n_players = len(self.players)
        if self.n_players != 2:
            raise ValueError("Only 2-player games supported currently")

    @classmethod
    def from_bimatrix(
        cls,
        payoffs_1: List[List[float]],
        payoffs_2: List[List[float]],
        strategy_names_1: Optional[List[str]] = None,
        strategy_names_2: Optional[List[str]] = None
    ) -> "NormalFormGame":
        """
        Create 2-player game from payoff matrices.

        Args:
            payoffs_1: Row player's payoff matrix
            payoffs_2: Column player's payoff matrix
        """
        n_strategies_1 = len(payoffs_1)
        n_strategies_2 = len(payoffs_1[0]) if payoffs_1 else 0

        if strategy_names_1 is None:
            strategy_names_1 = [f"S1_{i}" for i in range(n_strategies_1)]
        if strategy_names_2 is None:
            strategy_names_2 = [f"S2_{j}" for j in range(n_strategies_2)]

        player1 = Player(
            "Player 1",
            [Strategy(name, i) for i, name in enumerate(strategy_names_1)]
        )
        player2 = Player(
            "Player 2",
            [Strategy(name, i) for i, name in enumerate(strategy_names_2)]
        )

        return cls(
            players=[player1, player2],
            payoff_matrices={0: payoffs_1, 1: payoffs_2}
        )

    def payoff(self, player_idx: int, action_profile: Tuple[int, ...]) -> float:
        """Get payoff for player given action profile."""
        i, j = action_profile
        return self.payoff_matrices[player_idx][i][j]

    def best_response(self, player_idx: int, opponent_strategy: List[float]) -> int:
        """
        Compute best response to opponent's mixed strategy.

        Returns index of best pure strategy.
        """
        n_strategies = len(self.players[player_idx].strategies)
        n_opp = len(opponent_strategy)

        best_payoff = float('-inf')
        best_action = 0

        for i in range(n_strategies):
            expected_payoff = 0.0
            for j in range(n_opp):
                if player_idx == 0:
                    expected_payoff += opponent_strategy[j] * self.payoff_matrices[0][i][j]
                else:
                    expected_payoff += opponent_strategy[j] * self.payoff_matrices[1][j][i]

            if expected_payoff > best_payoff:
                best_payoff = expected_payoff
                best_action = i

        return best_action

    def is_nash_equilibrium(self, strategy_profile: Tuple[List[float], List[float]]) -> bool:
        """
        Check if strategy profile is Nash equilibrium.

        A profile is NE if no player can improve by deviating.
        """
        eps = 1e-8

        # Check player 1
        p1_payoffs = []
        for i in range(len(self.players[0].strategies)):
            payoff = sum(
                strategy_profile[1][j] * self.payoff_matrices[0][i][j]
                for j in range(len(self.players[1].strategies))
            )
            p1_payoffs.append(payoff)

        current_p1_payoff = sum(
            strategy_profile[0][i] * p1_payoffs[i]
            for i in range(len(p1_payoffs))
        )

        if max(p1_payoffs) > current_p1_payoff + eps:
            return False

        # Check player 2
        p2_payoffs = []
        for j in range(len(self.players[1].strategies)):
            payoff = sum(
                strategy_profile[0][i] * self.payoff_matrices[1][i][j]
                for i in range(len(self.players[0].strategies))
            )
            p2_payoffs.append(payoff)

        current_p2_payoff = sum(
            strategy_profile[1][j] * p2_payoffs[j]
            for j in range(len(p2_payoffs))
        )

        if max(p2_payoffs) > current_p2_payoff + eps:
            return False

        return True

    def find_pure_nash_equilibria(self) -> List[Tuple[int, int]]:
        """Find all pure strategy Nash equilibria."""
        equilibria = []
        n1 = len(self.players[0].strategies)
        n2 = len(self.players[1].strategies)

        for i in range(n1):
            for j in range(n2):
                # Check if (i, j) is NE
                # Player 1 best response to j
                p1_payoffs = [self.payoff_matrices[0][k][j] for k in range(n1)]
                if self.payoff_matrices[0][i][j] < max(p1_payoffs) - 1e-10:
                    continue

                # Player 2 best response to i
                p2_payoffs = [self.payoff_matrices[1][i][k] for k in range(n2)]
                if self.payoff_matrices[1][i][j] < max(p2_payoffs) - 1e-10:
                    continue

                equilibria.append((i, j))

        return equilibria

    def eliminate_dominated(self) -> "NormalFormGame":
        """
        Eliminate strictly dominated strategies.

        Returns new game with dominated strategies removed.
        """
        payoffs_1 = [row[:] for row in self.payoff_matrices[0]]
        payoffs_2 = [row[:] for row in self.payoff_matrices[1]]

        # Track remaining strategy indices
        remaining_1 = list(range(len(payoffs_1)))
        remaining_2 = list(range(len(payoffs_2[0])))

        changed = True
        while changed:
            changed = False

            # Check player 1 strategies
            to_remove = []
            for i in remaining_1:
                for i2 in remaining_1:
                    if i == i2:
                        continue
                    # Check if i is dominated by i2
                    dominated = all(
                        payoffs_1[i][j] < payoffs_1[i2][j]
                        for j in remaining_2
                    )
                    if dominated:
                        to_remove.append(i)
                        break

            for i in to_remove:
                remaining_1.remove(i)
                changed = True

            # Check player 2 strategies
            to_remove = []
            for j in remaining_2:
                for j2 in remaining_2:
                    if j == j2:
                        continue
                    # Check if j is dominated by j2
                    dominated = all(
                        payoffs_2[i][j] < payoffs_2[i][j2]
                        for i in remaining_1
                    )
                    if dominated:
                        to_remove.append(j)
                        break

            for j in to_remove:
                remaining_2.remove(j)
                changed = True

        # Build reduced game
        new_payoffs_1 = [
            [payoffs_1[i][j] for j in remaining_2]
            for i in remaining_1
        ]
        new_payoffs_2 = [
            [payoffs_2[i][j] for j in remaining_2]
            for i in remaining_1
        ]

        return NormalFormGame.from_bimatrix(new_payoffs_1, new_payoffs_2)


def support_enumeration_nash(game: NormalFormGame) -> List[Tuple[List[float], List[float]]]:
    """
    Find all Nash equilibria using support enumeration.

    For small games, enumerate all possible supports and check
    if they yield a valid equilibrium.

    Args:
        game: 2-player normal form game

    Returns:
        List of Nash equilibria as (mixed_strategy_1, mixed_strategy_2)
    """
    n1 = len(game.players[0].strategies)
    n2 = len(game.players[1].strategies)

    equilibria = []

    # Helper to generate subsets
    def subsets(n: int) -> List[List[int]]:
        result = []
        for i in range(1, 2 ** n):
            subset = [j for j in range(n) if (i >> j) & 1]
            result.append(subset)
        return result

    for support1 in subsets(n1):
        for support2 in subsets(n2):
            # Try to find equilibrium with these supports
            eq = _solve_support(game, support1, support2)
            if eq is not None:
                # Verify it's actually an equilibrium
                if game.is_nash_equilibrium(eq):
                    equilibria.append(eq)

    return equilibria


def _solve_support(
    game: NormalFormGame,
    support1: List[int],
    support2: List[int]
) -> Optional[Tuple[List[float], List[float]]]:
    """Solve for equilibrium with given supports."""
    n1 = len(game.players[0].strategies)
    n2 = len(game.players[1].strategies)
    k1 = len(support1)
    k2 = len(support2)

    if k1 == 0 or k2 == 0:
        return None

    # For valid equilibrium with support S:
    # 1. All strategies in S must yield same expected payoff
    # 2. Strategies outside S must yield lower payoff

    # Simple case: both supports are size 1 (pure NE)
    if k1 == 1 and k2 == 1:
        i, j = support1[0], support2[0]
        p1 = [0.0] * n1
        p2 = [0.0] * n2
        p1[i] = 1.0
        p2[j] = 1.0
        return (p1, p2)

    # For mixed strategies, solve system of equations
    # This is a simplified solver for 2x2 games

    if k1 == 2 and k2 == 2 and n1 == 2 and n2 == 2:
        # 2x2 game with full support
        a = game.payoff_matrices[1]

        # Player 2 indifferent: p1[0]*a[0][0] + p1[1]*a[1][0] = p1[0]*a[0][1] + p1[1]*a[1][1]
        # With p1[0] + p1[1] = 1
        # a[0][0]*p1[0] + a[1][0]*(1-p1[0]) = a[0][1]*p1[0] + a[1][1]*(1-p1[0])

        denom1 = (a[0][0] - a[0][1] - a[1][0] + a[1][1])
        if abs(denom1) < 1e-10:
            return None
        p1_0 = (a[1][1] - a[1][0]) / denom1

        if p1_0 < 0 or p1_0 > 1:
            return None

        b = game.payoff_matrices[0]
        denom2 = (b[0][0] - b[0][1] - b[1][0] + b[1][1])
        if abs(denom2) < 1e-10:
            return None
        p2_0 = (b[1][1] - b[0][1]) / denom2

        if p2_0 < 0 or p2_0 > 1:
            return None

        return ([p1_0, 1 - p1_0], [p2_0, 1 - p2_0])

    return None


def fictitious_play(
    game: NormalFormGame,
    iterations: int = 1000,
    seed: Optional[int] = None
) -> Tuple[List[float], List[float]]:
    """
    Find approximate Nash equilibrium using fictitious play.

    Players alternate best-responding to empirical distribution
    of opponent's past play.

    Args:
        game: 2-player normal form game
        iterations: Number of rounds
        seed: Unused. Fictitious play is deterministic; kept for API
            compatibility.

    Returns:
        Approximate mixed strategy Nash equilibrium
    """
    n1 = len(game.players[0].strategies)
    n2 = len(game.players[1].strategies)

    # Count frequencies
    counts_1 = [1.0] * n1  # Start with uniform prior
    counts_2 = [1.0] * n2

    for t in range(iterations):
        # Compute empirical frequencies
        total_1 = sum(counts_1)
        total_2 = sum(counts_2)
        freq_1 = [c / total_1 for c in counts_1]
        freq_2 = [c / total_2 for c in counts_2]

        # Best responses
        br_1 = game.best_response(0, freq_2)
        br_2 = game.best_response(1, freq_1)

        # Update counts
        counts_1[br_1] += 1
        counts_2[br_2] += 1

    # Final frequencies
    total_1 = sum(counts_1)
    total_2 = sum(counts_2)

    return (
        [c / total_1 for c in counts_1],
        [c / total_2 for c in counts_2]
    )


# =============================================================================
# Classic Games
# =============================================================================

def prisoners_dilemma() -> NormalFormGame:
    """
    Prisoner's Dilemma game.

    Payoffs: (C, C) = (3, 3), (C, D) = (0, 5), (D, C) = (5, 0), (D, D) = (1, 1)
    """
    return NormalFormGame.from_bimatrix(
        [[3, 0], [5, 1]],  # Player 1: Cooperate, Defect
        [[3, 5], [0, 1]],  # Player 2: Cooperate, Defect
        ["Cooperate", "Defect"],
        ["Cooperate", "Defect"]
    )


def matching_pennies() -> NormalFormGame:
    """
    Matching Pennies game.

    Zero-sum game with unique mixed Nash equilibrium (0.5, 0.5).
    """
    return NormalFormGame.from_bimatrix(
        [[1, -1], [-1, 1]],
        [[-1, 1], [1, -1]],
        ["Heads", "Tails"],
        ["Heads", "Tails"]
    )


def battle_of_sexes() -> NormalFormGame:
    """
    Battle of the Sexes game.

    Coordination game with two pure and one mixed NE.
    """
    return NormalFormGame.from_bimatrix(
        [[3, 0], [0, 2]],
        [[2, 0], [0, 3]],
        ["Opera", "Football"],
        ["Opera", "Football"]
    )


def stag_hunt() -> NormalFormGame:
    """
    Stag Hunt game.

    Coordination game illustrating social cooperation.
    """
    return NormalFormGame.from_bimatrix(
        [[4, 0], [3, 3]],
        [[4, 3], [0, 3]],
        ["Stag", "Hare"],
        ["Stag", "Hare"]
    )


def chicken() -> NormalFormGame:
    """
    Chicken (Hawk-Dove) game.

    Anti-coordination game with two pure and one mixed NE.
    """
    return NormalFormGame.from_bimatrix(
        [[0, -1], [1, -10]],
        [[0, 1], [-1, -10]],
        ["Swerve", "Straight"],
        ["Swerve", "Straight"]
    )
