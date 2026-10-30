"""
Lookback Options

Implements lookback options where payoff depends on the maximum
or minimum price achieved during the option's life:

- Floating Strike Lookback Call: max(S_T - S_min, 0)
- Floating Strike Lookback Put: max(S_max - S_T, 0)
- Fixed Strike Lookback Call: max(S_max - K, 0)
- Fixed Strike Lookback Put: max(K - S_min, 0)

Closed-form solutions available for continuous monitoring (Goldman-Sosin-Gatto 1979).
Discrete monitoring requires Monte Carlo or continuity correction.

References:
- Goldman, Sosin, Gatto (1979): Early lookback formulas
- Conze & Viswanathan (1991): Complete analysis
- Broadie, Glasserman, Kou (1999): Discrete monitoring correction
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math

from finceptquantlib.core.foundations.types import OptionType
from finceptquantlib.core.utilities.distributions import norm_cdf, norm_pdf


@dataclass
class LookbackOption:
    """
    Base class for lookback options.

    Attributes:
        spot: Current spot price
        T: Time to maturity
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        option_type: CALL or PUT
        S_max: Current running maximum (for puts, fixed strike calls)
        S_min: Current running minimum (for calls, fixed strike puts)
    """
    spot: float
    T: float
    r: float
    q: float = 0.0
    sigma: float = 0.2
    option_type: OptionType = OptionType.CALL
    S_max: Optional[float] = None
    S_min: Optional[float] = None

    def __post_init__(self):
        # Initialize running extremes if not provided
        if self.S_max is None:
            self.S_max = self.spot
        if self.S_min is None:
            self.S_min = self.spot

        # Validate
        if self.S_max < self.spot:
            raise ValueError("S_max must be >= spot")
        if self.S_min > self.spot:
            raise ValueError("S_min must be <= spot")


class FloatingStrikeLookback(LookbackOption):
    """
    Floating Strike Lookback Option.

    The strike is set at expiry based on the path:
    - Call payoff: S_T - S_min (buy at the minimum)
    - Put payoff: S_max - S_T (sell at the maximum)

    These are always in-the-money (unless all prices equal).

    Closed-form (Goldman-Sosin-Gatto):
    For call with q != r:
    C = S × exp(-qT) × N(a1) - S_min × exp(-rT) × N(a1 - σ√T)
        + S × exp(-rT) × (σ²/(2(r-q))) × [...]

    For call with q = r:
    C = S × exp(-qT) × [N(a1) - (σ²T/2) × N(-a1) + σ√T × n(a1)]
        - S_min × exp(-rT) × N(a1 - σ√T)
    """

    def analytical_price(self) -> float:
        """
        Goldman-Sosin-Gatto closed-form for floating strike lookback.

        Returns:
            Option price
        """
        S = self.spot
        T = self.T
        r = self.r
        q = self.q
        sigma = self.sigma

        if T <= 0:
            if self.option_type == OptionType.CALL:
                return max(S - self.S_min, 0)
            else:
                return max(self.S_max - S, 0)

        if self.option_type == OptionType.CALL:
            M = self.S_min
            return self._floating_call(S, M, T, r, q, sigma)
        else:
            M = self.S_max
            return self._floating_put(S, M, T, r, q, sigma)

    def _floating_call(
        self,
        S: float,
        M: float,  # S_min
        T: float,
        r: float,
        q: float,
        sigma: float
    ) -> float:
        """Floating strike lookback call."""
        sqrt_T = math.sqrt(T)
        sigma_sqrt_T = sigma * sqrt_T

        if abs(sigma_sqrt_T) < 1e-10:
            return S * math.exp(-q * T) - M * math.exp(-r * T)

        a1 = (math.log(S / M) + (r - q + 0.5 * sigma ** 2) * T) / sigma_sqrt_T
        a2 = a1 - sigma_sqrt_T

        if abs(r - q) < 1e-10:
            # q = r case
            term1 = S * math.exp(-q * T) * norm_cdf(a1)
            term2 = M * math.exp(-r * T) * norm_cdf(a2)
            term3 = S * math.exp(-q * T) * sigma_sqrt_T * (
                norm_pdf(a1) + a1 * norm_cdf(a1)
            )
            return term1 - term2 + term3
        else:
            # General case
            term1 = S * math.exp(-q * T) * norm_cdf(a1)
            term2 = M * math.exp(-r * T) * norm_cdf(a2)

            # Correction term
            term3 = S * math.exp(-r * T) * sigma ** 2 / (2 * (r - q)) * (
                (S / M) ** (-2 * (r - q) / sigma ** 2) * norm_cdf(a1 - 2 * (r - q) * sqrt_T / sigma)
                - math.exp((r - q) * T) * norm_cdf(a1)
            )

            return term1 - term2 - term3

    def _floating_put(
        self,
        S: float,
        M: float,  # S_max
        T: float,
        r: float,
        q: float,
        sigma: float
    ) -> float:
        """Floating strike lookback put."""
        sqrt_T = math.sqrt(T)
        sigma_sqrt_T = sigma * sqrt_T

        if abs(sigma_sqrt_T) < 1e-10:
            return M * math.exp(-r * T) - S * math.exp(-q * T)

        b1 = (math.log(S / M) + (r - q + 0.5 * sigma ** 2) * T) / sigma_sqrt_T
        b2 = b1 - sigma_sqrt_T

        if abs(r - q) < 1e-10:
            # q = r case
            term1 = M * math.exp(-r * T) * norm_cdf(-b2)
            term2 = S * math.exp(-q * T) * norm_cdf(-b1)
            term3 = S * math.exp(-q * T) * sigma_sqrt_T * (
                norm_pdf(b1) + b1 * norm_cdf(b1)
            )
            return term1 - term2 + term3
        else:
            # General case
            term1 = M * math.exp(-r * T) * norm_cdf(-b2)
            term2 = S * math.exp(-q * T) * norm_cdf(-b1)

            term3 = S * math.exp(-r * T) * sigma ** 2 / (2 * (r - q)) * (
                -(S / M) ** (-2 * (r - q) / sigma ** 2) * norm_cdf(-b1 + 2 * (r - q) * sqrt_T / sigma)
                + math.exp((r - q) * T) * norm_cdf(-b1)
            )

            return term1 - term2 - term3

    def monte_carlo_price(
        self,
        n_paths: int = 100000,
        n_steps: int = 252,
        seed: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Monte Carlo pricing for floating strike lookback.

        Args:
            n_paths: Number of simulation paths
            n_steps: Time steps
            seed: Random seed

        Returns:
            (price, standard error)
        """
        import random
        rng = random.Random(seed)

        dt = self.T / n_steps
        drift = (self.r - self.q - 0.5 * self.sigma ** 2) * dt
        vol = self.sigma * math.sqrt(dt)

        payoffs = []

        for _ in range(n_paths):
            S_t = self.spot
            S_min = min(self.S_min, self.spot)
            S_max = max(self.S_max, self.spot)

            for _ in range(n_steps):
                Z = rng.gauss(0, 1)
                S_t *= math.exp(drift + vol * Z)
                S_min = min(S_min, S_t)
                S_max = max(S_max, S_t)

            if self.option_type == OptionType.CALL:
                payoff = S_t - S_min
            else:
                payoff = S_max - S_t

            payoffs.append(payoff)

        price = math.exp(-self.r * self.T) * sum(payoffs) / n_paths
        variance = sum((p - sum(payoffs) / n_paths) ** 2 for p in payoffs) / (n_paths - 1)
        std_error = math.exp(-self.r * self.T) * math.sqrt(variance / n_paths)

        return price, std_error


@dataclass
class FixedStrikeLookback(LookbackOption):
    """
    Fixed Strike Lookback Option.

    The strike is fixed, but payoff is based on the extreme:
    - Call payoff: max(S_max - K, 0)
    - Put payoff: max(K - S_min, 0)

    These can be out-of-the-money if the extreme never reaches the strike.

    Attributes:
        strike: Fixed strike price
    """
    strike: float = 100.0

    def analytical_price(self) -> float:
        """
        Conze-Viswanathan closed-form for fixed strike lookback.

        Returns:
            Option price
        """
        S = self.spot
        K = self.strike
        T = self.T
        r = self.r
        q = self.q
        sigma = self.sigma

        if T <= 0:
            if self.option_type == OptionType.CALL:
                return max(self.S_max - K, 0)
            else:
                return max(K - self.S_min, 0)

        if self.option_type == OptionType.CALL:
            M = self.S_max  # Running maximum
            return self._fixed_call(S, K, M, T, r, q, sigma)
        else:
            M = self.S_min  # Running minimum
            return self._fixed_put(S, K, M, T, r, q, sigma)

    def _fixed_call(
        self,
        S: float,
        K: float,
        M: float,  # S_max
        T: float,
        r: float,
        q: float,
        sigma: float
    ) -> float:
        """Fixed strike lookback call: max(S_max - K, 0)."""
        sqrt_T = math.sqrt(T)
        sigma_sqrt_T = sigma * sqrt_T

        if sigma_sqrt_T < 1e-10:
            # Zero vol
            S_T = S * math.exp((r - q) * T)
            return math.exp(-r * T) * max(max(M, S_T) - K, 0)

        # Case 1: M >= K (already in the money)
        if M >= K:
            # Option is in the money, payoff = M - K + lookback on (K to S_max)
            e1 = (math.log(S / M) + (r - q + 0.5 * sigma ** 2) * T) / sigma_sqrt_T
            e2 = e1 - sigma_sqrt_T

            if abs(r - q) < 1e-10:
                term1 = (M - K) * math.exp(-r * T)
                term2 = S * math.exp(-q * T) * norm_cdf(e1)
                term3 = M * math.exp(-r * T) * norm_cdf(e2)
                term4 = S * math.exp(-r * T) * sigma ** 2 * T / 2 * norm_cdf(-e1)
                return term1 + term2 - term3 + term4
            else:
                lambda_ = (r - q) / sigma ** 2
                term1 = (M - K) * math.exp(-r * T)
                term2 = S * math.exp(-q * T) * norm_cdf(e1)
                term3 = M * math.exp(-r * T) * norm_cdf(e2)
                term4 = S * math.exp(-r * T) / (2 * lambda_) * (
                    (S / M) ** (-2 * lambda_) * norm_cdf(e1 - 2 * lambda_ * sigma_sqrt_T)
                    - math.exp((r - q) * T) * norm_cdf(e1)
                )
                return term1 + term2 - term3 - term4

        else:  # M < K
            # Not yet in the money
            d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / sigma_sqrt_T
            d2 = d1 - sigma_sqrt_T

            e1 = (math.log(S / M) + (r - q + 0.5 * sigma ** 2) * T) / sigma_sqrt_T
            e2 = e1 - sigma_sqrt_T

            if abs(r - q) < 1e-10:
                term1 = S * math.exp(-q * T) * (norm_cdf(d1) - norm_cdf(e1))
                term2 = K * math.exp(-r * T) * norm_cdf(d2)
                term3 = M * math.exp(-r * T) * norm_cdf(e2)
                term4 = S * math.exp(-r * T) * sigma ** 2 * T / 2 * (
                    norm_cdf(-d1) - norm_cdf(-e1)
                )
                return term1 - term2 + term3 + term4
            else:
                lambda_ = (r - q) / sigma ** 2
                term1 = S * math.exp(-q * T) * norm_cdf(d1)
                term2 = K * math.exp(-r * T) * norm_cdf(d2)

                # Lookback premium
                term3 = S * math.exp(-r * T) / (2 * lambda_) * (
                    -(S / K) ** (-2 * lambda_) * norm_cdf(d1 - 2 * lambda_ * sigma_sqrt_T)
                    + math.exp((r - q) * T) * norm_cdf(d1)
                )

                return term1 - term2 - term3

    def _fixed_put(
        self,
        S: float,
        K: float,
        M: float,  # S_min
        T: float,
        r: float,
        q: float,
        sigma: float
    ) -> float:
        """Fixed strike lookback put: max(K - S_min, 0)."""
        sqrt_T = math.sqrt(T)
        sigma_sqrt_T = sigma * sqrt_T

        if sigma_sqrt_T < 1e-10:
            S_T = S * math.exp((r - q) * T)
            return math.exp(-r * T) * max(K - min(M, S_T), 0)

        # Case 1: M <= K (already in the money)
        if M <= K:
            f1 = (math.log(S / M) + (r - q + 0.5 * sigma ** 2) * T) / sigma_sqrt_T
            f2 = f1 - sigma_sqrt_T

            if abs(r - q) < 1e-10:
                term1 = (K - M) * math.exp(-r * T)
                term2 = M * math.exp(-r * T) * norm_cdf(-f2)
                term3 = S * math.exp(-q * T) * norm_cdf(-f1)
                term4 = S * math.exp(-r * T) * sigma ** 2 * T / 2 * norm_cdf(f1)
                return term1 + term2 - term3 + term4
            else:
                lambda_ = (r - q) / sigma ** 2
                term1 = (K - M) * math.exp(-r * T)
                term2 = M * math.exp(-r * T) * norm_cdf(-f2)
                term3 = S * math.exp(-q * T) * norm_cdf(-f1)
                term4 = S * math.exp(-r * T) / (2 * lambda_) * (
                    (S / M) ** (-2 * lambda_) * norm_cdf(-f1 + 2 * lambda_ * sigma_sqrt_T)
                    - math.exp((r - q) * T) * norm_cdf(-f1)
                )
                return term1 + term2 - term3 + term4

        else:  # M > K
            d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / sigma_sqrt_T
            d2 = d1 - sigma_sqrt_T

            if abs(r - q) < 1e-10:
                term1 = K * math.exp(-r * T) * norm_cdf(-d2)
                term2 = S * math.exp(-q * T) * norm_cdf(-d1)
                term3 = S * math.exp(-r * T) * sigma ** 2 * T / 2 * norm_cdf(d1)
                return term1 - term2 + term3
            else:
                lambda_ = (r - q) / sigma ** 2
                term1 = K * math.exp(-r * T) * norm_cdf(-d2)
                term2 = S * math.exp(-q * T) * norm_cdf(-d1)
                term3 = S * math.exp(-r * T) / (2 * lambda_) * (
                    (S / K) ** (-2 * lambda_) * norm_cdf(-d1 + 2 * lambda_ * sigma_sqrt_T)
                    - math.exp((r - q) * T) * norm_cdf(-d1)
                )
                return term1 - term2 + term3

    def monte_carlo_price(
        self,
        n_paths: int = 100000,
        n_steps: int = 252,
        seed: Optional[int] = None
    ) -> Tuple[float, float]:
        """Monte Carlo pricing for fixed strike lookback."""
        import random
        rng = random.Random(seed)

        dt = self.T / n_steps
        drift = (self.r - self.q - 0.5 * self.sigma ** 2) * dt
        vol = self.sigma * math.sqrt(dt)

        payoffs = []

        for _ in range(n_paths):
            S_t = self.spot
            S_min = min(self.S_min, self.spot)
            S_max = max(self.S_max, self.spot)

            for _ in range(n_steps):
                Z = rng.gauss(0, 1)
                S_t *= math.exp(drift + vol * Z)
                S_min = min(S_min, S_t)
                S_max = max(S_max, S_t)

            if self.option_type == OptionType.CALL:
                payoff = max(S_max - self.strike, 0)
            else:
                payoff = max(self.strike - S_min, 0)

            payoffs.append(payoff)

        price = math.exp(-self.r * self.T) * sum(payoffs) / n_paths
        variance = sum((p - sum(payoffs) / n_paths) ** 2 for p in payoffs) / (n_paths - 1)
        std_error = math.exp(-self.r * self.T) * math.sqrt(variance / n_paths)

        return price, std_error


# Convenience functions

def floating_lookback_call(
    spot: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    S_min: Optional[float] = None
) -> float:
    """Price a floating strike lookback call."""
    opt = FloatingStrikeLookback(
        spot=spot,
        T=T,
        r=r,
        q=q,
        sigma=sigma,
        option_type=OptionType.CALL,
        S_min=S_min if S_min else spot
    )
    return opt.analytical_price()


def floating_lookback_put(
    spot: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    S_max: Optional[float] = None
) -> float:
    """Price a floating strike lookback put."""
    opt = FloatingStrikeLookback(
        spot=spot,
        T=T,
        r=r,
        q=q,
        sigma=sigma,
        option_type=OptionType.PUT,
        S_max=S_max if S_max else spot
    )
    return opt.analytical_price()


def fixed_lookback_call(
    spot: float,
    strike: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    S_max: Optional[float] = None
) -> float:
    """Price a fixed strike lookback call."""
    opt = FixedStrikeLookback(
        spot=spot,
        strike=strike,
        T=T,
        r=r,
        q=q,
        sigma=sigma,
        option_type=OptionType.CALL,
        S_max=S_max if S_max else spot
    )
    return opt.analytical_price()


def fixed_lookback_put(
    spot: float,
    strike: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    S_min: Optional[float] = None
) -> float:
    """Price a fixed strike lookback put."""
    opt = FixedStrikeLookback(
        spot=spot,
        strike=strike,
        T=T,
        r=r,
        q=q,
        sigma=sigma,
        option_type=OptionType.PUT,
        S_min=S_min if S_min else spot
    )
    return opt.analytical_price()
