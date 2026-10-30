"""
Economics Module

Mathematical economics and decision theory:

Utility Theory (economics.utility):
- CRRA, CARA, Log, Quadratic utility functions
- Prospect Theory value function
- Risk aversion measures
- Certainty equivalents and risk premia
- Stochastic dominance tests

Game Theory (economics.game_theory):
- Normal form games and Nash equilibrium
- Dominated strategy elimination
- Classic games (Prisoner's Dilemma, etc.)
- Auction theory (first/second price, all-pay)
- Revenue equivalence

Equilibrium (economics.equilibrium):
- Exchange economies
- Walrasian equilibrium
- Contract curves
- Cobb-Douglas and CES preferences

Example:
    >>> from finceptquantlib.economics import CRRA, prisoners_dilemma
    >>>
    >>> # Risk-averse agent
    >>> u = CRRA(gamma=3)
    >>> u.certainty_equivalent([90, 110])
    98.508682
    >>>
    >>> # Game theory
    >>> game = prisoners_dilemma()
    >>> game.find_pure_nash_equilibria()
    [(1, 1)]
"""

from finceptquantlib.economics.utility import (
    UtilityFunction,
    CRRA,
    CARA,
    QuadraticUtility,
    LogUtility,
    ProspectTheoryValue,
    HyperbolicAbsoluteRiskAversion,
    expected_utility,
    certainty_equivalent_approximation,
    stochastic_dominance_first,
    stochastic_dominance_second,
)

from finceptquantlib.economics.game_theory import (
    Strategy,
    Player,
    NormalFormGame,
    support_enumeration_nash,
    fictitious_play,
    prisoners_dilemma,
    matching_pennies,
    battle_of_sexes,
    stag_hunt,
    chicken,
    AuctionFormat,
    FirstPriceAuction,
    SecondPriceAuction,
    AllPayAuction,
    revenue_equivalence_theorem,
    optimal_reserve_price,
)

from finceptquantlib.economics.equilibrium import (
    Consumer,
    ExchangeEconomy,
    find_walrasian_equilibrium,
    contract_curve,
    CobbDouglasConsumer,
    CESConsumer,
)

__all__ = [
    # Utility
    "UtilityFunction",
    "CRRA",
    "CARA",
    "QuadraticUtility",
    "LogUtility",
    "ProspectTheoryValue",
    "HyperbolicAbsoluteRiskAversion",
    "expected_utility",
    "certainty_equivalent_approximation",
    "stochastic_dominance_first",
    "stochastic_dominance_second",
    # Game Theory
    "Strategy",
    "Player",
    "NormalFormGame",
    "support_enumeration_nash",
    "fictitious_play",
    "prisoners_dilemma",
    "matching_pennies",
    "battle_of_sexes",
    "stag_hunt",
    "chicken",
    "AuctionFormat",
    "FirstPriceAuction",
    "SecondPriceAuction",
    "AllPayAuction",
    "revenue_equivalence_theorem",
    "optimal_reserve_price",
    # Equilibrium
    "Consumer",
    "ExchangeEconomy",
    "find_walrasian_equilibrium",
    "contract_curve",
    "CobbDouglasConsumer",
    "CESConsumer",
]
