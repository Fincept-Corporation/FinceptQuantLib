"""
Barrier Options

Implements various barrier option types:
- Single barriers: Up-and-in, up-and-out, down-and-in, down-and-out
- Double barriers: Knock-in and knock-out with two barriers
- Discrete barriers: Monitoring at specific dates
- Rebates: Payments upon knock-out

Pricing methods:
- Analytical formulas (continuous monitoring)
- Monte Carlo simulation
- PDE methods (finite differences)

References:
- Merton (1973): Early barrier option analysis
- Rubinstein & Reiner (1991): Analytical formulas
- Hui (1996): Double barrier formulas
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import math

from finceptquantlib.core.foundations.types import OptionType, BarrierType
from finceptquantlib.core.utilities.distributions import norm_cdf


@dataclass
class BarrierOption:
    """
    Single Barrier Option.

    A barrier option is a path-dependent option where the payoff depends
    on whether the underlying price reaches a certain barrier level.

    Attributes:
        spot: Current spot price
        strike: Strike price
        barrier: Barrier level
        T: Time to maturity (years)
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        option_type: CALL or PUT
        barrier_type: Type of barrier (up/down, in/out)
        rebate: Amount paid if option knocks out (default 0)
    """
    spot: float
    strike: float
    barrier: float
    T: float
    r: float
    q: float = 0.0
    sigma: float = 0.2
    option_type: OptionType = OptionType.CALL
    barrier_type: BarrierType = BarrierType.DOWN_AND_OUT
    rebate: float = 0.0

    def __post_init__(self):
        self._validate()

    def _validate(self):
        """Validate barrier option parameters."""
        if self.spot <= 0:
            raise ValueError("Spot price must be positive")
        if self.strike <= 0:
            raise ValueError("Strike must be positive")
        if self.barrier <= 0:
            raise ValueError("Barrier must be positive")
        if self.T <= 0:
            raise ValueError("Time to maturity must be positive")
        if self.sigma <= 0:
            raise ValueError("Volatility must be positive")

        # Check barrier position
        if self.barrier_type in (BarrierType.DOWN_AND_IN, BarrierType.DOWN_AND_OUT):
            if self.barrier >= self.spot:
                raise ValueError("Down barrier must be below spot price")
        else:
            if self.barrier <= self.spot:
                raise ValueError("Up barrier must be above spot price")

    def analytical_price(self) -> float:
        """
        Price using Rubinstein-Reiner (1991) analytical formulas.

        The formulas use combinations of Black-Scholes type terms
        with adjustments for the barrier.

        Returns:
            Option price
        """
        S = self.spot
        K = self.strike
        H = self.barrier  # Barrier
        T = self.T
        r = self.r
        q = self.q
        sigma = self.sigma
        R = self.rebate

        # Helper parameters
        mu = (r - q - 0.5 * sigma ** 2) / sigma ** 2
        lambda_ = math.sqrt(mu ** 2 + 2 * r / sigma ** 2)

        sqrt_T = math.sqrt(T)

        # x, y parameters
        x1 = math.log(S / K) / (sigma * sqrt_T) + (1 + mu) * sigma * sqrt_T
        x2 = math.log(S / H) / (sigma * sqrt_T) + (1 + mu) * sigma * sqrt_T
        y1 = math.log(H ** 2 / (S * K)) / (sigma * sqrt_T) + (1 + mu) * sigma * sqrt_T
        y2 = math.log(H / S) / (sigma * sqrt_T) + (1 + mu) * sigma * sqrt_T

        # z for rebate
        z = math.log(H / S) / (sigma * sqrt_T) + lambda_ * sigma * sqrt_T

        # phi terms (+1 for call, -1 for put)
        phi = 1 if self.option_type == OptionType.CALL else -1

        # eta terms (+1 for down, -1 for up barriers)
        eta = 1 if self.barrier_type in (BarrierType.DOWN_AND_IN, BarrierType.DOWN_AND_OUT) else -1

        # Build formula components
        A = phi * S * math.exp(-q * T) * norm_cdf(phi * x1) - \
            phi * K * math.exp(-r * T) * norm_cdf(phi * x1 - phi * sigma * sqrt_T)

        B = phi * S * math.exp(-q * T) * norm_cdf(phi * x2) - \
            phi * K * math.exp(-r * T) * norm_cdf(phi * x2 - phi * sigma * sqrt_T)

        C = phi * S * math.exp(-q * T) * (H / S) ** (2 * (mu + 1)) * norm_cdf(eta * y1) - \
            phi * K * math.exp(-r * T) * (H / S) ** (2 * mu) * norm_cdf(eta * y1 - eta * sigma * sqrt_T)

        D = phi * S * math.exp(-q * T) * (H / S) ** (2 * (mu + 1)) * norm_cdf(eta * y2) - \
            phi * K * math.exp(-r * T) * (H / S) ** (2 * mu) * norm_cdf(eta * y2 - eta * sigma * sqrt_T)

        # Rebate terms
        E = R * math.exp(-r * T) * (
            norm_cdf(eta * x2 - eta * sigma * sqrt_T) -
            (H / S) ** (2 * mu) * norm_cdf(eta * y2 - eta * sigma * sqrt_T)
        )

        F = R * (
            (H / S) ** (mu + lambda_) * norm_cdf(eta * z) +
            (H / S) ** (mu - lambda_) * norm_cdf(eta * z - 2 * eta * lambda_ * sigma * sqrt_T)
        )

        # Combine based on barrier type
        if self.barrier_type == BarrierType.DOWN_AND_IN:
            if self.option_type == OptionType.CALL:
                if K > H:
                    return C + E
                else:
                    return A - B + D + E
            else:  # PUT
                if K > H:
                    return B - C + D + E
                else:
                    return A + E

        elif self.barrier_type == BarrierType.DOWN_AND_OUT:
            if self.option_type == OptionType.CALL:
                if K > H:
                    return A - C + F
                else:
                    return B - D + F
            else:  # PUT
                if K > H:
                    return A - B + C - D + F
                else:
                    return F

        elif self.barrier_type == BarrierType.UP_AND_IN:
            if self.option_type == OptionType.CALL:
                if K > H:
                    return A + E
                else:
                    return B - C + D + E
            else:  # PUT
                if K > H:
                    return C + E
                else:
                    return A - B + D + E

        elif self.barrier_type == BarrierType.UP_AND_OUT:
            if self.option_type == OptionType.CALL:
                if K > H:
                    return F
                else:
                    return A - B + C - D + F
            else:  # PUT
                if K > H:
                    return A - C + F
                else:
                    return B - D + F

        return 0.0

    def monte_carlo_price(
        self,
        n_paths: int = 100000,
        n_steps: int = 252,
        seed: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Price barrier option using Monte Carlo simulation.

        Args:
            n_paths: Number of simulation paths
            n_steps: Number of time steps
            seed: Random seed

        Returns:
            Tuple of (price, standard error)
        """
        import random
        rng = random.Random(seed)

        S = self.spot
        K = self.strike
        H = self.barrier
        T = self.T
        r = self.r
        q = self.q
        sigma = self.sigma
        R = self.rebate

        dt = T / n_steps
        drift = (r - q - 0.5 * sigma ** 2) * dt
        vol_sqrt_dt = sigma * math.sqrt(dt)

        is_down = self.barrier_type in (BarrierType.DOWN_AND_IN, BarrierType.DOWN_AND_OUT)
        is_knock_out = self.barrier_type in (BarrierType.DOWN_AND_OUT, BarrierType.UP_AND_OUT)

        payoffs = []

        for _ in range(n_paths):
            S_t = S
            knocked = False

            for _ in range(n_steps):
                Z = rng.gauss(0, 1)
                S_t *= math.exp(drift + vol_sqrt_dt * Z)

                # Check barrier
                if is_down and S_t <= H:
                    knocked = True
                    break
                elif not is_down and S_t >= H:
                    knocked = True
                    break

            # Calculate payoff
            if is_knock_out:
                if knocked:
                    payoff = R  # Rebate
                else:
                    if self.option_type == OptionType.CALL:
                        payoff = max(S_t - K, 0)
                    else:
                        payoff = max(K - S_t, 0)
            else:  # Knock-in
                if knocked:
                    if self.option_type == OptionType.CALL:
                        payoff = max(S_t - K, 0)
                    else:
                        payoff = max(K - S_t, 0)
                else:
                    payoff = R  # Rebate for not knocking in

            payoffs.append(payoff)

        # Statistics
        price = math.exp(-r * T) * sum(payoffs) / n_paths
        variance = sum((p - sum(payoffs) / n_paths) ** 2 for p in payoffs) / (n_paths - 1)
        std_error = math.exp(-r * T) * math.sqrt(variance / n_paths)

        return price, std_error

    def delta(self, bump: float = 0.01) -> float:
        """Calculate delta via finite difference."""
        S = self.spot
        up = BarrierOption(
            S * (1 + bump), self.strike, self.barrier, self.T,
            self.r, self.q, self.sigma, self.option_type, self.barrier_type
        )
        down = BarrierOption(
            S * (1 - bump), self.strike, self.barrier, self.T,
            self.r, self.q, self.sigma, self.option_type, self.barrier_type
        )
        return (up.analytical_price() - down.analytical_price()) / (2 * bump * S)

    def gamma(self, bump: float = 0.01) -> float:
        """Calculate gamma via finite difference."""
        S = self.spot
        up = BarrierOption(
            S * (1 + bump), self.strike, self.barrier, self.T,
            self.r, self.q, self.sigma, self.option_type, self.barrier_type
        )
        down = BarrierOption(
            S * (1 - bump), self.strike, self.barrier, self.T,
            self.r, self.q, self.sigma, self.option_type, self.barrier_type
        )
        center = self.analytical_price()
        return (up.analytical_price() - 2 * center + down.analytical_price()) / (bump * S) ** 2

    def vega(self, bump: float = 0.01) -> float:
        """Calculate vega via finite difference."""
        up = BarrierOption(
            self.spot, self.strike, self.barrier, self.T,
            self.r, self.q, self.sigma + bump, self.option_type, self.barrier_type
        )
        down = BarrierOption(
            self.spot, self.strike, self.barrier, self.T,
            self.r, self.q, self.sigma - bump, self.option_type, self.barrier_type
        )
        return (up.analytical_price() - down.analytical_price()) / (2 * bump)


@dataclass
class DoubleBarrierOption:
    """
    Double Barrier Option.

    An option with both upper and lower barriers. Can be either:
    - Knock-out: Option becomes worthless if either barrier is hit
    - Knock-in: Option activates only if a barrier is hit

    Uses Hui (1996) analytical formulas.
    """
    spot: float
    strike: float
    lower_barrier: float
    upper_barrier: float
    T: float
    r: float
    q: float = 0.0
    sigma: float = 0.2
    option_type: OptionType = OptionType.CALL
    knock_out: bool = True  # True for knock-out, False for knock-in

    def __post_init__(self):
        if self.lower_barrier >= self.upper_barrier:
            raise ValueError("Lower barrier must be less than upper barrier")
        if not (self.lower_barrier < self.spot < self.upper_barrier):
            raise ValueError("Spot must be between barriers")

    def analytical_price(self, n_terms: int = 50) -> float:
        """
        Price double barrier option using infinite series formula.

        The knock-out double barrier call/put uses Ikeda & Kunitomo (1992)
        or Hui (1996) formulas.

        Args:
            n_terms: Number of series terms

        Returns:
            Option price
        """
        S = self.spot
        K = self.strike
        L = self.lower_barrier
        U = self.upper_barrier
        T = self.T
        r = self.r
        q = self.q
        sigma = self.sigma

        # Parameters
        mu = (r - q) / sigma ** 2 - 0.5
        b = math.log(U / L)

        # Series summation for knock-out
        price = 0.0

        for n in range(-n_terms, n_terms + 1):
            # d_n parameters
            x_n = math.log(S / L) + 2 * n * b

            # First term
            d1_n = (x_n + (mu + 1) * sigma ** 2 * T) / (sigma * math.sqrt(T))
            d2_n = d1_n - sigma * math.sqrt(T)

            # Standard barrier adjustments
            factor = (L / S) ** (2 * n)

            if self.option_type == OptionType.CALL:
                term = factor * (
                    S * math.exp(-q * T) * norm_cdf(d1_n) -
                    K * math.exp(-r * T) * norm_cdf(d2_n)
                )
            else:
                term = factor * (
                    K * math.exp(-r * T) * norm_cdf(-d2_n) -
                    S * math.exp(-q * T) * norm_cdf(-d1_n)
                )

            price += term

            # Check convergence
            if abs(term) < 1e-12 and abs(n) > 5:
                break

        price = max(price, 0)

        if not self.knock_out:
            # Knock-in = vanilla - knock-out
            vanilla = self._vanilla_price()
            price = vanilla - price

        return price

    def monte_carlo_price(
        self,
        n_paths: int = 100000,
        n_steps: int = 252,
        seed: Optional[int] = None
    ) -> Tuple[float, float]:
        """Monte Carlo pricing for double barrier option."""
        import random
        rng = random.Random(seed)

        dt = self.T / n_steps
        drift = (self.r - self.q - 0.5 * self.sigma ** 2) * dt
        vol_sqrt_dt = self.sigma * math.sqrt(dt)

        payoffs = []

        for _ in range(n_paths):
            S_t = self.spot
            knocked = False

            for _ in range(n_steps):
                Z = rng.gauss(0, 1)
                S_t *= math.exp(drift + vol_sqrt_dt * Z)

                if S_t <= self.lower_barrier or S_t >= self.upper_barrier:
                    knocked = True
                    break

            if self.knock_out:
                if knocked:
                    payoff = 0
                else:
                    if self.option_type == OptionType.CALL:
                        payoff = max(S_t - self.strike, 0)
                    else:
                        payoff = max(self.strike - S_t, 0)
            else:
                if knocked:
                    if self.option_type == OptionType.CALL:
                        payoff = max(S_t - self.strike, 0)
                    else:
                        payoff = max(self.strike - S_t, 0)
                else:
                    payoff = 0

            payoffs.append(payoff)

        price = math.exp(-self.r * self.T) * sum(payoffs) / n_paths
        variance = sum((p - sum(payoffs) / n_paths) ** 2 for p in payoffs) / (n_paths - 1)
        std_error = math.exp(-self.r * self.T) * math.sqrt(variance / n_paths)

        return price, std_error

    def _vanilla_price(self) -> float:
        """Calculate vanilla option price using Black-Scholes."""
        S, K, T, r, q, sigma = self.spot, self.strike, self.T, self.r, self.q, self.sigma

        d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if self.option_type == OptionType.CALL:
            return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
        else:
            return K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)


@dataclass
class DiscreteBarrierOption:
    """
    Barrier Option with Discrete Monitoring.

    The barrier is checked only at specific dates, not continuously.
    This is priced via Monte Carlo or using Broadie-Glasserman-Kou
    continuity correction.

    The continuity correction adjusts the barrier:
    - For down barriers: H_adj = H × exp(β × σ × √Δt)
    - For up barriers: H_adj = H × exp(-β × σ × √Δt)

    where β ≈ 0.5826 (related to ζ(1/2)/√(2π))
    """
    spot: float
    strike: float
    barrier: float
    T: float
    r: float
    q: float = 0.0
    sigma: float = 0.2
    option_type: OptionType = OptionType.CALL
    barrier_type: BarrierType = BarrierType.DOWN_AND_OUT
    monitoring_dates: Optional[List[float]] = None  # As year fractions
    n_monitoring: int = 252  # Default daily monitoring

    def __post_init__(self):
        if self.monitoring_dates is None:
            # Create evenly spaced monitoring dates
            self.monitoring_dates = [
                self.T * (i + 1) / self.n_monitoring
                for i in range(self.n_monitoring)
            ]

    def price_with_correction(self) -> float:
        """
        Price using continuity correction (Broadie-Glasserman-Kou).

        Adjusts the barrier based on monitoring frequency to approximate
        discrete monitoring with a continuous barrier formula.
        """
        # Broadie-Glasserman-Kou correction
        BETA = 0.5826  # ζ(1/2) / √(2π)

        # Average monitoring interval
        n = len(self.monitoring_dates)
        delta_t = self.T / n

        # Adjusted barrier
        is_down = self.barrier_type in (BarrierType.DOWN_AND_IN, BarrierType.DOWN_AND_OUT)

        if is_down:
            H_adj = self.barrier * math.exp(BETA * self.sigma * math.sqrt(delta_t))
        else:
            H_adj = self.barrier * math.exp(-BETA * self.sigma * math.sqrt(delta_t))

        # Price with adjusted barrier using continuous formula
        adjusted_option = BarrierOption(
            self.spot, self.strike, H_adj, self.T,
            self.r, self.q, self.sigma, self.option_type, self.barrier_type
        )

        return adjusted_option.analytical_price()

    def monte_carlo_price(
        self,
        n_paths: int = 100000,
        seed: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Monte Carlo pricing with exact discrete monitoring.

        Simulates paths and checks barrier only at monitoring dates.
        """
        import random
        rng = random.Random(seed)

        is_down = self.barrier_type in (BarrierType.DOWN_AND_IN, BarrierType.DOWN_AND_OUT)
        is_knock_out = self.barrier_type in (BarrierType.DOWN_AND_OUT, BarrierType.UP_AND_OUT)

        payoffs = []
        dates = [0.0] + self.monitoring_dates

        for _ in range(n_paths):
            S_t = self.spot
            knocked = False

            for i in range(1, len(dates)):
                dt = dates[i] - dates[i - 1]

                # Simulate to next monitoring date
                drift = (self.r - self.q - 0.5 * self.sigma ** 2) * dt
                vol = self.sigma * math.sqrt(dt)
                Z = rng.gauss(0, 1)

                S_t *= math.exp(drift + vol * Z)

                # Check barrier at monitoring date
                if is_down and S_t <= self.barrier:
                    knocked = True
                    if is_knock_out:
                        break
                elif not is_down and S_t >= self.barrier:
                    knocked = True
                    if is_knock_out:
                        break

            # Calculate payoff
            if is_knock_out:
                if knocked:
                    payoff = 0
                else:
                    if self.option_type == OptionType.CALL:
                        payoff = max(S_t - self.strike, 0)
                    else:
                        payoff = max(self.strike - S_t, 0)
            else:  # Knock-in
                if knocked:
                    if self.option_type == OptionType.CALL:
                        payoff = max(S_t - self.strike, 0)
                    else:
                        payoff = max(self.strike - S_t, 0)
                else:
                    payoff = 0

            payoffs.append(payoff)

        price = math.exp(-self.r * self.T) * sum(payoffs) / n_paths
        variance = sum((p - sum(payoffs) / n_paths) ** 2 for p in payoffs) / (n_paths - 1)
        std_error = math.exp(-self.r * self.T) * math.sqrt(variance / n_paths)

        return price, std_error


# Convenience functions

def down_and_out_call(
    spot: float,
    strike: float,
    barrier: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    rebate: float = 0.0
) -> float:
    """Price a down-and-out call option."""
    opt = BarrierOption(
        spot, strike, barrier, T, r, q, sigma,
        OptionType.CALL, BarrierType.DOWN_AND_OUT, rebate
    )
    return opt.analytical_price()


def up_and_out_call(
    spot: float,
    strike: float,
    barrier: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    rebate: float = 0.0
) -> float:
    """Price an up-and-out call option."""
    opt = BarrierOption(
        spot, strike, barrier, T, r, q, sigma,
        OptionType.CALL, BarrierType.UP_AND_OUT, rebate
    )
    return opt.analytical_price()


def down_and_in_put(
    spot: float,
    strike: float,
    barrier: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    rebate: float = 0.0
) -> float:
    """Price a down-and-in put option."""
    opt = BarrierOption(
        spot, strike, barrier, T, r, q, sigma,
        OptionType.PUT, BarrierType.DOWN_AND_IN, rebate
    )
    return opt.analytical_price()
