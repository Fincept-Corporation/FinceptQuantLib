"""
Game Theory

Strategic interaction and mechanism design:

Normal Form Games:
- NormalFormGame: General 2-player games
- Pure and mixed Nash equilibrium computation
- Dominated strategy elimination
- Classic games: Prisoner's Dilemma, Matching Pennies, etc.

Auction Theory:
- FirstPriceAuction: Sealed bid, pay your bid
- SecondPriceAuction: Vickrey auction
- AllPayAuction: Everyone pays
- Revenue equivalence analysis

Example:
    >>> from finceptquantlib.economics.game_theory import (
    ...     prisoners_dilemma, SecondPriceAuction
    ... )
    >>>
    >>> # Nash equilibrium in Prisoner's Dilemma
    >>> game = prisoners_dilemma()
    >>> game.find_pure_nash_equilibria()  # [(1, 1)] - both defect
    [(1, 1)]
    >>>
    >>> # Vickrey auction
    >>> auction = SecondPriceAuction(n_bidders=5)
    >>> result = auction.run_auction([0.3, 0.5, 0.7, 0.4, 0.6])
"""

from finceptquantlib.economics.game_theory.games import (
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
)

from finceptquantlib.economics.game_theory.auctions import (
    AuctionFormat,
    BidResult,
    AuctionResult,
    FirstPriceAuction,
    SecondPriceAuction,
    AllPayAuction,
    revenue_equivalence_theorem,
    optimal_reserve_price,
    simulate_auctions,
)

__all__ = [
    # Games
    "Strategy",
    "Player",
    "NormalFormGame",
    "support_enumeration_nash",
    "fictitious_play",
    # Classic games
    "prisoners_dilemma",
    "matching_pennies",
    "battle_of_sexes",
    "stag_hunt",
    "chicken",
    # Auctions
    "AuctionFormat",
    "BidResult",
    "AuctionResult",
    "FirstPriceAuction",
    "SecondPriceAuction",
    "AllPayAuction",
    "revenue_equivalence_theorem",
    "optimal_reserve_price",
    "simulate_auctions",
]
