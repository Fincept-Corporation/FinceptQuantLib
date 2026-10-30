"""
Auction Theory

Common auction formats and equilibrium analysis:
- First-price sealed bid
- Second-price (Vickrey)
- English (ascending)
- Dutch (descending)
- Revenue equivalence
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple


class AuctionFormat(Enum):
    """Auction format types."""
    FIRST_PRICE = "first_price"
    SECOND_PRICE = "second_price"
    ENGLISH = "english"
    DUTCH = "dutch"
    ALL_PAY = "all_pay"


@dataclass
class BidResult:
    """Result of a bidder's participation."""
    bidder_id: int
    valuation: float
    bid: float
    won: bool
    payment: float
    surplus: float


@dataclass
class AuctionResult:
    """Result of an auction."""
    format: AuctionFormat
    winning_bidder: int
    winning_bid: float
    revenue: float
    bids: List[BidResult]


@dataclass
class FirstPriceAuction:
    """
    First-price sealed bid auction.

    Highest bidder wins and pays their bid.

    In equilibrium with n bidders and uniform [0,1] valuations:
    b(v) = (n-1)/n * v
    """
    n_bidders: int
    reserve_price: float = 0.0

    def equilibrium_bid(self, valuation: float) -> float:
        """
        Compute equilibrium bid for symmetric IPV model.

        b(v) = v - ∫₀ᵛ F(x)^(n-1) dx / F(v)^(n-1)

        For uniform [0,1]: b(v) = (n-1)/n * v
        """
        if valuation <= self.reserve_price:
            return 0.0
        n = self.n_bidders
        # Uniform [0,1] case
        return (n - 1) / n * valuation

    def expected_revenue(self, cdf: Callable[[float], float] = lambda x: x) -> float:
        """
        Expected revenue in equilibrium.

        For uniform [0,1]: E[Revenue] = (n-1)/(n+1)
        """
        n = self.n_bidders
        # Uniform case
        return (n - 1) / (n + 1)

    def run_auction(
        self,
        valuations: List[float],
        bids: Optional[List[float]] = None
    ) -> AuctionResult:
        """
        Run auction with given valuations and bids.

        If bids not provided, uses equilibrium bids.
        """
        if bids is None:
            bids = [self.equilibrium_bid(v) for v in valuations]

        # Find winner
        valid_bids = [(b, i) for i, b in enumerate(bids) if b >= self.reserve_price]

        if not valid_bids:
            # No valid bids
            return AuctionResult(
                format=AuctionFormat.FIRST_PRICE,
                winning_bidder=-1,
                winning_bid=0.0,
                revenue=0.0,
                bids=[
                    BidResult(i, valuations[i], bids[i], False, 0.0, 0.0)
                    for i in range(len(valuations))
                ]
            )

        winner_bid, winner_idx = max(valid_bids)

        results = []
        for i in range(len(valuations)):
            won = (i == winner_idx)
            payment = winner_bid if won else 0.0
            surplus = valuations[i] - payment if won else 0.0
            results.append(BidResult(i, valuations[i], bids[i], won, payment, surplus))

        return AuctionResult(
            format=AuctionFormat.FIRST_PRICE,
            winning_bidder=winner_idx,
            winning_bid=winner_bid,
            revenue=winner_bid,
            bids=results
        )


@dataclass
class SecondPriceAuction:
    """
    Second-price (Vickrey) auction.

    Highest bidder wins and pays second-highest bid.

    Dominant strategy: bid your valuation.
    """
    n_bidders: int
    reserve_price: float = 0.0

    def equilibrium_bid(self, valuation: float) -> float:
        """
        Dominant strategy bid.

        In second-price auction, bidding true valuation is
        weakly dominant regardless of other bidders' strategies.
        """
        return valuation

    def expected_revenue(self) -> float:
        """
        Expected revenue for uniform [0,1] valuations.

        E[Revenue] = (n-1)/(n+1) (same as first-price by revenue equivalence)
        """
        n = self.n_bidders
        return (n - 1) / (n + 1)

    def run_auction(
        self,
        valuations: List[float],
        bids: Optional[List[float]] = None
    ) -> AuctionResult:
        """Run auction with given valuations and bids."""
        if bids is None:
            bids = [self.equilibrium_bid(v) for v in valuations]

        # Find winner and second price
        indexed_bids = sorted(
            [(b, i) for i, b in enumerate(bids) if b >= self.reserve_price],
            reverse=True
        )

        if not indexed_bids:
            return AuctionResult(
                format=AuctionFormat.SECOND_PRICE,
                winning_bidder=-1,
                winning_bid=0.0,
                revenue=0.0,
                bids=[
                    BidResult(i, valuations[i], bids[i], False, 0.0, 0.0)
                    for i in range(len(valuations))
                ]
            )

        winner_bid, winner_idx = indexed_bids[0]
        second_price = indexed_bids[1][0] if len(indexed_bids) > 1 else self.reserve_price

        results = []
        for i in range(len(valuations)):
            won = (i == winner_idx)
            payment = second_price if won else 0.0
            surplus = valuations[i] - payment if won else 0.0
            results.append(BidResult(i, valuations[i], bids[i], won, payment, surplus))

        return AuctionResult(
            format=AuctionFormat.SECOND_PRICE,
            winning_bidder=winner_idx,
            winning_bid=winner_bid,
            revenue=second_price,
            bids=results
        )


@dataclass
class AllPayAuction:
    """
    All-pay auction.

    Highest bidder wins, but all bidders pay their bids.

    Equilibrium (symmetric, uniform [0,1]):
    b(v) = (n-1)/n * v^n
    """
    n_bidders: int

    def equilibrium_bid(self, valuation: float) -> float:
        """
        Compute equilibrium bid for symmetric IPV model.

        b(v) = (n-1)/n * v^n for uniform [0,1]
        """
        n = self.n_bidders
        return (n - 1) / n * (valuation ** n)

    def expected_revenue(self) -> float:
        """
        Expected revenue.

        All bidders pay, so revenue = n * E[b(v)]
        For uniform: E[Revenue] = (n-1)/(n+1)
        """
        n = self.n_bidders
        return (n - 1) / (n + 1)

    def run_auction(
        self,
        valuations: List[float],
        bids: Optional[List[float]] = None
    ) -> AuctionResult:
        """Run all-pay auction."""
        if bids is None:
            bids = [self.equilibrium_bid(v) for v in valuations]

        # Find winner
        winner_idx = max(range(len(bids)), key=lambda i: bids[i])
        total_revenue = sum(bids)

        results = []
        for i in range(len(valuations)):
            won = (i == winner_idx)
            payment = bids[i]
            surplus = valuations[i] - payment if won else -payment
            results.append(BidResult(i, valuations[i], bids[i], won, payment, surplus))

        return AuctionResult(
            format=AuctionFormat.ALL_PAY,
            winning_bidder=winner_idx,
            winning_bid=bids[winner_idx],
            revenue=total_revenue,
            bids=results
        )


def revenue_equivalence_theorem(
    n_bidders: int,
    cdf: Callable[[float], float],
    pdf: Callable[[float], float],
    lower: float = 0.0,
    upper: float = 1.0,
    n_points: int = 100
) -> float:
    """
    Compute expected revenue using Revenue Equivalence Theorem.

    Under symmetric IPV with risk-neutral bidders:
    E[Revenue] = E[v₍ₙ₋₁₎] = ∫ v * g(v) dv

    where g is pdf of second-highest order statistic.

    Args:
        n_bidders: Number of bidders
        cdf: CDF of value distribution
        pdf: PDF of value distribution
        lower, upper: Support of distribution
        n_points: Integration points

    Returns:
        Expected revenue
    """
    n = n_bidders
    h = (upper - lower) / n_points
    revenue = 0.0

    for i in range(1, n_points):
        v = lower + i * h

        # PDF of second-highest order statistic
        # g(v) = n(n-1) * f(v) * F(v)^(n-2) * (1-F(v))
        F_v = cdf(v)
        f_v = pdf(v)

        g_v = n * (n - 1) * f_v * (F_v ** (n - 2)) * (1 - F_v)

        revenue += v * g_v * h

    return revenue


def optimal_reserve_price(
    n_bidders: int,
    cdf: Callable[[float], float],
    pdf: Callable[[float], float],
    lower: float = 0.0,
    upper: float = 1.0,
    seller_value: float = 0.0
) -> float:
    """
    Compute optimal reserve price (Myerson).

    r* = v₀ + (1 - F(r*)) / f(r*)

    where v₀ is seller's value.

    For uniform [0,1] with v₀=0: r* = 0.5
    """
    # Solve r = v₀ + (1 - F(r)) / f(r) using bisection
    r_low, r_high = lower, upper

    for _ in range(50):
        r = (r_low + r_high) / 2
        f_r = pdf(r)

        if f_r < 1e-10:
            r_high = r
            continue

        virtual_value = r - (1 - cdf(r)) / f_r

        if virtual_value < seller_value:
            r_low = r
        else:
            r_high = r

        if abs(r_high - r_low) < 1e-8:
            break

    return (r_low + r_high) / 2


def simulate_auctions(
    format: AuctionFormat,
    n_bidders: int,
    n_simulations: int = 10000,
    value_dist: str = "uniform",
    seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Simulate auctions and compute average revenue.

    Args:
        format: Auction format
        n_bidders: Number of bidders
        n_simulations: Number of simulations
        value_dist: Value distribution ("uniform" or "exponential")
        seed: Random seed

    Returns:
        (mean_revenue, std_revenue)
    """
    rng = random.Random(seed)

    revenues = []

    for _ in range(n_simulations):
        # Draw valuations
        if value_dist == "uniform":
            valuations = [rng.random() for _ in range(n_bidders)]
        elif value_dist == "exponential":
            valuations = [rng.expovariate(1) for _ in range(n_bidders)]
        else:
            valuations = [rng.random() for _ in range(n_bidders)]

        # Run auction
        if format == AuctionFormat.FIRST_PRICE:
            auction = FirstPriceAuction(n_bidders)
        elif format == AuctionFormat.SECOND_PRICE:
            auction = SecondPriceAuction(n_bidders)
        elif format == AuctionFormat.ALL_PAY:
            auction = AllPayAuction(n_bidders)
        else:
            auction = SecondPriceAuction(n_bidders)

        result = auction.run_auction(valuations)
        revenues.append(result.revenue)

    mean_rev = sum(revenues) / len(revenues)
    var_rev = sum((r - mean_rev) ** 2 for r in revenues) / len(revenues)

    return mean_rev, math.sqrt(var_rev)
