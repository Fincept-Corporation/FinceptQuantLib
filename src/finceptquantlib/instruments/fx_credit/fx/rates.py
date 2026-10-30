"""
FX Rate System

Multi-currency FX rate management with automatic triangulation.

Features:
- Store spot rates between currencies
- Automatic triangulation for cross rates
- Bid/ask spread handling
- Optimal path finding for conversions
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Set, Union
from collections import defaultdict

from finceptquantlib.core.foundations.types import (
    Currency
)


# =============================================================================
# Quote Conventions
# =============================================================================

class QuoteConvention(Enum):
    """FX quote convention."""
    DIRECT = auto()       # Units of quote per 1 base (EUR/USD = 1.08)
    INDIRECT = auto()     # Units of base per 1 quote
    PERCENTAGE = auto()   # Rate as percentage (rarely used)


# =============================================================================
# FX Quote
# =============================================================================

@dataclass
class FXQuote:
    """
    FX rate quote with optional bid/ask spread.

    Attributes:
        base: Base currency (numerator)
        quote: Quote currency (denominator)
        mid: Mid rate
        bid: Bid rate (optional)
        ask: Ask rate (optional)
        value_date: Value date for the rate
    """
    base: Currency
    quote: Currency
    mid: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    value_date: Optional[date] = None
    source: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.base, str):
            self.base = Currency.from_string(self.base)
        if isinstance(self.quote, str):
            self.quote = Currency.from_string(self.quote)

        # Validate rates
        if self.mid <= 0:
            raise ValueError(f"Mid rate must be positive: {self.mid}")
        if self.bid is not None and self.bid <= 0:
            raise ValueError(f"Bid rate must be positive: {self.bid}")
        if self.ask is not None and self.ask <= 0:
            raise ValueError(f"Ask rate must be positive: {self.ask}")
        if self.bid is not None and self.ask is not None and self.bid > self.ask:
            raise ValueError(f"Bid ({self.bid}) must be <= Ask ({self.ask})")

    @property
    def pair(self) -> str:
        """Currency pair string."""
        return f"{self.base.value}/{self.quote.value}"

    @property
    def spread(self) -> Optional[float]:
        """Bid-ask spread in absolute terms."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None

    @property
    def spread_pct(self) -> Optional[float]:
        """Bid-ask spread as percentage of mid."""
        if self.spread is not None:
            return self.spread / self.mid * 100
        return None

    @property
    def spread_pips(self) -> Optional[float]:
        """Bid-ask spread in pips (for major pairs)."""
        if self.spread is not None:
            # For JPY pairs, 1 pip = 0.01; for others, 1 pip = 0.0001
            pip_size = 0.01 if self.quote == Currency.JPY else 0.0001
            return self.spread / pip_size
        return None

    def invert(self) -> FXQuote:
        """Return inverted quote (swap base and quote)."""
        new_mid = 1.0 / self.mid
        new_bid = 1.0 / self.ask if self.ask else None
        new_ask = 1.0 / self.bid if self.bid else None

        return FXQuote(
            base=self.quote,
            quote=self.base,
            mid=new_mid,
            bid=new_bid,
            ask=new_ask,
            value_date=self.value_date,
            source=self.source
        )

    def convert(self, amount: float, direction: str = "base_to_quote") -> float:
        """
        Convert amount between currencies.

        Args:
            amount: Amount to convert
            direction: "base_to_quote" or "quote_to_base"

        Returns:
            Converted amount
        """
        if direction == "base_to_quote":
            return amount * self.mid
        elif direction == "quote_to_base":
            return amount / self.mid
        else:
            raise ValueError(f"Invalid direction: {direction}")

    def __repr__(self) -> str:
        if self.bid and self.ask:
            return f"FXQuote({self.pair}: {self.bid:.5f}/{self.ask:.5f})"
        return f"FXQuote({self.pair}: {self.mid:.5f})"


# =============================================================================
# FX Rate Matrix
# =============================================================================

class FXRateMatrix:
    """
    Multi-currency FX rate matrix with automatic triangulation.

    Stores spot rates and computes cross rates through optimal paths.
    Supports bid/ask spreads with proper handling during triangulation.

    Example:
        >>> fx = FXRateMatrix()
        >>> fx.add("EUR/USD", 1.0850)
        >>> fx.add("USD/JPY", 149.50)
        >>> fx.get("EUR/JPY")  # Auto-triangulates
        162.2075

        >>> fx.convert(1_000_000, "EUR", "JPY")
        162207500.0
    """

    def __init__(self, value_date: Optional[date] = None) -> None:
        """
        Initialize FX rate matrix.

        Args:
            value_date: Default value date for rates
        """
        self.value_date = value_date
        self._quotes: Dict[str, FXQuote] = {}
        self._graph: Dict[Currency, Set[Currency]] = defaultdict(set)

    def add(
        self,
        pair: str,
        rate: Union[float, FXQuote],
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        source: str = ""
    ) -> None:
        """
        Add an FX rate to the matrix.

        Args:
            pair: Currency pair "XXX/YYY" (e.g., "EUR/USD")
            rate: Mid rate or FXQuote object
            bid: Optional bid rate
            ask: Optional ask rate
            source: Rate source
        """
        if isinstance(rate, FXQuote):
            quote = rate
        else:
            base_str, quote_str = pair.upper().split("/")
            base = Currency.from_string(base_str)
            quote_ccy = Currency.from_string(quote_str)

            quote = FXQuote(
                base=base,
                quote=quote_ccy,
                mid=rate,
                bid=bid,
                ask=ask,
                value_date=self.value_date,
                source=source
            )

        # Store the quote
        self._quotes[quote.pair] = quote

        # Update the graph for triangulation
        self._graph[quote.base].add(quote.quote)
        self._graph[quote.quote].add(quote.base)

    def get(
        self,
        pair: str,
        include_spread: bool = False
    ) -> Union[float, FXQuote]:
        """
        Get FX rate for a currency pair.

        Triangulates if direct rate not available.

        Args:
            pair: Currency pair "XXX/YYY"
            include_spread: If True, return FXQuote with bid/ask

        Returns:
            Mid rate (float) or FXQuote if include_spread=True
        """
        pair = pair.upper()

        # Check for direct quote
        if pair in self._quotes:
            quote = self._quotes[pair]
            return quote if include_spread else quote.mid

        # Check for inverted quote
        base_str, quote_str = pair.split("/")
        inverted_pair = f"{quote_str}/{base_str}"
        if inverted_pair in self._quotes:
            quote = self._quotes[inverted_pair].invert()
            return quote if include_spread else quote.mid

        # Try triangulation
        base = Currency.from_string(base_str)
        quote_ccy = Currency.from_string(quote_str)

        triangulated = self._triangulate(base, quote_ccy)
        if triangulated is not None:
            return triangulated if include_spread else triangulated.mid

        raise ValueError(f"Cannot determine rate for {pair}")

    def _triangulate(
        self,
        base: Currency,
        quote: Currency
    ) -> Optional[FXQuote]:
        """Triangulate cross rate through intermediate currency."""
        # BFS to find path
        path = self._find_path(base, quote)
        if path is None:
            return None

        # Calculate cross rate along path
        rate = 1.0
        bid_rate = 1.0
        ask_rate = 1.0
        has_spread = True

        for i in range(len(path) - 1):
            curr_base = path[i]
            curr_quote = path[i + 1]
            pair = f"{curr_base.value}/{curr_quote.value}"
            inv_pair = f"{curr_quote.value}/{curr_base.value}"

            if pair in self._quotes:
                q = self._quotes[pair]
                rate *= q.mid
                if q.bid and q.ask:
                    bid_rate *= q.bid
                    ask_rate *= q.ask
                else:
                    has_spread = False
            elif inv_pair in self._quotes:
                q = self._quotes[inv_pair]
                rate /= q.mid
                if q.bid and q.ask:
                    bid_rate /= q.ask
                    ask_rate /= q.bid
                else:
                    has_spread = False
            else:
                return None

        return FXQuote(
            base=base,
            quote=quote,
            mid=rate,
            bid=bid_rate if has_spread else None,
            ask=ask_rate if has_spread else None,
            value_date=self.value_date
        )

    def _find_path(
        self,
        start: Currency,
        end: Currency
    ) -> Optional[List[Currency]]:
        """Find shortest path between currencies using BFS."""
        if start == end:
            return [start]

        visited = {start}
        queue = [[start]]

        while queue:
            path = queue.pop(0)
            current = path[-1]

            for neighbor in self._graph[current]:
                if neighbor == end:
                    return path + [end]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])

        return None

    def convert(
        self,
        amount: float,
        from_ccy: Union[str, Currency],
        to_ccy: Union[str, Currency],
        side: str = "mid"
    ) -> float:
        """
        Convert amount between currencies.

        Args:
            amount: Amount to convert
            from_ccy: Source currency
            to_ccy: Target currency
            side: "mid", "bid", or "ask"

        Returns:
            Converted amount
        """
        if isinstance(from_ccy, str):
            from_ccy = Currency.from_string(from_ccy)
        if isinstance(to_ccy, str):
            to_ccy = Currency.from_string(to_ccy)

        if from_ccy == to_ccy:
            return amount

        pair = f"{from_ccy.value}/{to_ccy.value}"
        quote = self.get(pair, include_spread=True)

        if isinstance(quote, FXQuote):
            if side == "mid":
                return amount * quote.mid
            elif side == "bid":
                return amount * (quote.bid or quote.mid)
            elif side == "ask":
                return amount * (quote.ask or quote.mid)

        return amount * float(quote)

    def pairs(self) -> List[str]:
        """List all available direct pairs."""
        return list(self._quotes.keys())

    def currencies(self) -> Set[Currency]:
        """Set of all currencies in the matrix."""
        return set(self._graph.keys())

    def __contains__(self, pair: str) -> bool:
        """Check if a pair (direct or triangulated) is available."""
        try:
            self.get(pair)
            return True
        except ValueError:
            return False


# =============================================================================
# Utility Functions
# =============================================================================

def parse_pair(pair: str) -> Tuple[Currency, Currency]:
    """Parse currency pair string into base and quote currencies."""
    pair = pair.upper().strip()
    if "/" in pair:
        base_str, quote_str = pair.split("/")
    elif len(pair) == 6:
        base_str, quote_str = pair[:3], pair[3:]
    else:
        raise ValueError(f"Invalid pair format: {pair}")

    return Currency.from_string(base_str), Currency.from_string(quote_str)


def invert_pair(pair: str) -> str:
    """Invert a currency pair string."""
    base, quote = parse_pair(pair)
    return f"{quote.value}/{base.value}"


def get_pip_size(pair: str) -> float:
    """Get pip size for a currency pair."""
    _, quote = parse_pair(pair)
    # JPY pairs have pip = 0.01, others have pip = 0.0001
    return 0.01 if quote == Currency.JPY else 0.0001


def rate_to_pips(rate: float, pair: str) -> float:
    """Convert rate to pips."""
    return rate / get_pip_size(pair)


def pips_to_rate(pips: float, pair: str) -> float:
    """Convert pips to rate."""
    return pips * get_pip_size(pair)
