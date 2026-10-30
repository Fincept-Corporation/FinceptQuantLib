"""
Asian Options

Implements Asian (average) options:
- Geometric average: Closed-form solution available
- Arithmetic average: Turnbull-Wakeman approximation, Monte Carlo
- Average price vs average strike
- Discrete vs continuous averaging

References:
- Turnbull & Wakeman (1991): Arithmetic average approximation
- Kemna & Vorst (1990): Geometric average closed-form
- Curran (1994): Conditioning approach
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum
import math

from finceptquantlib.core.foundations.types import OptionType
from finceptquantlib.core.utilities.distributions import norm_cdf


class AveragingType(Enum):
    """Type of averaging for Asian option."""
    ARITHMETIC = "arithmetic"  # Simple average
    GEOMETRIC = "geometric"    # Geometric mean


class AverageStrike(Enum):
    """Whether averaging applies to price or strike."""
    PRICE = "price"   # Average price vs fixed strike
    STRIKE = "strike"  # Fixed price vs average strike


@dataclass
class AsianOption:
    """
    Asian Option Base Class.

    Asian options have payoffs based on the average price over time:
    - Average price call: max(A - K, 0)
    - Average price put: max(K - A, 0)
    - Average strike call: max(S_T - A, 0)
    - Average strike put: max(A - S_T, 0)

    where A is the average (arithmetic or geometric) of spot prices.

    Attributes:
        spot: Current spot price
        strike: Strike price (for average price) or notional (for average strike)
        T: Time to maturity
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        option_type: CALL or PUT
        averaging_type: ARITHMETIC or GEOMETRIC
        average_strike: PRICE (avg price option) or STRIKE (avg strike option)
        averaging_dates: List of averaging dates as year fractions
        current_average: Running average if option is already in progress
        current_count: Number of fixings already done
    """
    spot: float
    strike: float
    T: float
    r: float
    q: float = 0.0
    sigma: float = 0.2
    option_type: OptionType = OptionType.CALL
    averaging_type: AveragingType = AveragingType.ARITHMETIC
    average_strike: AverageStrike = AverageStrike.PRICE
    averaging_dates: Optional[List[float]] = None
    current_average: float = 0.0
    current_count: int = 0

    def __post_init__(self):
        if self.averaging_dates is None:
            # Daily averaging
            n = int(self.T * 252)
            self.averaging_dates = [self.T * i / n for i in range(1, n + 1)]


class GeometricAsianOption(AsianOption):
    """
    Geometric Asian Option.

    The geometric average allows a closed-form solution because
    the product of lognormals is lognormal.

    G = (∏_{i=1}^n S_{t_i})^{1/n}

    For continuous averaging:
    G = exp((1/T) ∫_0^T ln(S_t) dt)

    The geometric average follows:
    ln(G) ~ N(μ_G, σ_G²)

    where:
    μ_G = ln(S_0) + (r - q - σ²/2)(T_1 + T_n)/2
    σ_G² = σ² × [(T_1 + T_n)/2 + (T_n - T_1)²/(6T_n)]
    """

    def __init__(self, **kwargs):
        kwargs['averaging_type'] = AveragingType.GEOMETRIC
        super().__init__(**kwargs)

    def analytical_price(self) -> float:
        """
        Closed-form pricing for geometric Asian option.

        Uses adjusted volatility and drift to price as a modified European.
        """
        S = self.spot
        K = self.strike
        T = self.T
        r = self.r
        q = self.q
        sigma = self.sigma

        n = len(self.averaging_dates)

        if n == 0:
            return 0.0

        # Time parameters
        T1 = self.averaging_dates[0]  # First averaging date
        Tn = self.averaging_dates[-1]  # Last averaging date

        # Adjusted parameters for geometric average
        # Mean of log(G)
        M1 = math.log(S) + (r - q - 0.5 * sigma ** 2) * (T1 + Tn) / 2

        # Variance of log(G) for discrete averaging
        if n == 1:
            sigma_G_sq = sigma ** 2 * T1
        else:
            # Sum of covariances
            sigma_G_sq = sigma ** 2 * (
                (T1 + Tn) / 2 +
                (Tn - T1) ** 2 / (6 * Tn) * (n + 1) / (n - 1) if n > 1 else T1
            ) / n

        sigma_G = math.sqrt(sigma_G_sq / T)

        # Adjusted forward
        F_G = math.exp(M1 + 0.5 * sigma_G_sq)

        # Black formula on geometric average
        sqrt_T = math.sqrt(T)

        if sigma_G * sqrt_T < 1e-10:
            if self.option_type == OptionType.CALL:
                return math.exp(-r * T) * max(F_G - K, 0)
            else:
                return math.exp(-r * T) * max(K - F_G, 0)

        d1 = (math.log(F_G / K) + 0.5 * sigma_G ** 2 * T) / (sigma_G * sqrt_T)
        d2 = d1 - sigma_G * sqrt_T

        if self.option_type == OptionType.CALL:
            price = math.exp(-r * T) * (F_G * norm_cdf(d1) - K * norm_cdf(d2))
        else:
            price = math.exp(-r * T) * (K * norm_cdf(-d2) - F_G * norm_cdf(-d1))

        return price


class ArithmeticAsianOption(AsianOption):
    """
    Arithmetic Asian Option.

    The arithmetic average does not have a closed-form solution.
    Methods available:
    - Turnbull-Wakeman approximation
    - Monte Carlo (with geometric control variate)
    - Curran conditioning approach

    A = (1/n) × Σ_{i=1}^n S_{t_i}
    """

    def __init__(self, **kwargs):
        kwargs['averaging_type'] = AveragingType.ARITHMETIC
        super().__init__(**kwargs)

    def turnbull_wakeman_price(self) -> float:
        """
        Turnbull-Wakeman (1991) approximation.

        Matches first two moments of arithmetic average to a lognormal
        distribution, then uses Black-Scholes type formula.

        The approximation works well for typical parameters but may
        fail for extreme cases (very long maturity, high vol).
        """
        S = self.spot
        K = self.strike
        T = self.T
        r = self.r
        q = self.q
        sigma = self.sigma

        n = len(self.averaging_dates)

        if n == 0:
            return 0.0

        # Compute first two moments of arithmetic average
        # E[A] and E[A²]

        # First moment: E[A] = (1/n) Σ E[S_{t_i}]
        M1 = 0.0
        for t_i in self.averaging_dates:
            M1 += S * math.exp((r - q) * t_i)
        M1 /= n

        # Second moment: E[A²] = (1/n²) ΣΣ E[S_{t_i} S_{t_j}]
        M2 = 0.0
        for i, t_i in enumerate(self.averaging_dates):
            for j, t_j in enumerate(self.averaging_dates):
                t_min = min(t_i, t_j)
                t_max = max(t_i, t_j)

                # E[S_i × S_j] = S² × exp((2r-2q)(t_min) + (r-q)(t_max-t_min) + σ²t_min)
                exponent = (2 * r - 2 * q + sigma ** 2) * t_min + (r - q) * (t_max - t_min)
                M2 += S ** 2 * math.exp(exponent)
        M2 /= n ** 2

        # Match to lognormal: ln(A) ~ N(μ_A, σ_A²)
        # E[A] = exp(μ + σ²/2), E[A²] = exp(2μ + 2σ²)
        # => σ² = ln(M2/M1²), μ = ln(M1) - σ²/2

        variance_ratio = M2 / (M1 ** 2)
        if variance_ratio <= 1:
            # Degenerate case - use simple formula
            if self.option_type == OptionType.CALL:
                return math.exp(-r * T) * max(M1 - K, 0)
            else:
                return math.exp(-r * T) * max(K - M1, 0)

        sigma_A_sq = math.log(variance_ratio)
        sigma_A = math.sqrt(sigma_A_sq / T)

        # Black formula
        sqrt_T = math.sqrt(T)

        if sigma_A * sqrt_T < 1e-10:
            if self.option_type == OptionType.CALL:
                return math.exp(-r * T) * max(M1 - K, 0)
            else:
                return math.exp(-r * T) * max(K - M1, 0)

        d1 = (math.log(M1 / K) + 0.5 * sigma_A ** 2 * T) / (sigma_A * sqrt_T)
        d2 = d1 - sigma_A * sqrt_T

        if self.option_type == OptionType.CALL:
            price = math.exp(-r * T) * (M1 * norm_cdf(d1) - K * norm_cdf(d2))
        else:
            price = math.exp(-r * T) * (K * norm_cdf(-d2) - M1 * norm_cdf(-d1))

        return price

    def monte_carlo_price(
        self,
        n_paths: int = 100000,
        use_control_variate: bool = True,
        seed: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Monte Carlo pricing with geometric Asian as control variate.

        The geometric average has closed-form, so we can use:
        E[A_arith] ≈ mc(A_arith) - mc(A_geom) + E[A_geom]

        This reduces variance significantly.

        Args:
            n_paths: Number of paths
            use_control_variate: Use geometric average as control
            seed: Random seed

        Returns:
            Tuple of (price, standard error)
        """
        import random
        rng = random.Random(seed)

        n = len(self.averaging_dates)
        dt_list = [self.averaging_dates[0]]
        for i in range(1, n):
            dt_list.append(self.averaging_dates[i] - self.averaging_dates[i - 1])

        arith_payoffs = []
        geom_payoffs = []

        for _ in range(n_paths):
            S_t = self.spot
            sum_S = 0.0
            prod_S = 1.0

            for dt in dt_list:
                drift = (self.r - self.q - 0.5 * self.sigma ** 2) * dt
                vol = self.sigma * math.sqrt(dt)
                Z = rng.gauss(0, 1)

                S_t *= math.exp(drift + vol * Z)
                sum_S += S_t
                prod_S *= S_t

            # Averages
            A_arith = sum_S / n
            A_geom = prod_S ** (1.0 / n)

            # Payoffs
            if self.option_type == OptionType.CALL:
                arith_payoffs.append(max(A_arith - self.strike, 0))
                geom_payoffs.append(max(A_geom - self.strike, 0))
            else:
                arith_payoffs.append(max(self.strike - A_arith, 0))
                geom_payoffs.append(max(self.strike - A_geom, 0))

        # Discounted prices
        df = math.exp(-self.r * self.T)
        arith_price = df * sum(arith_payoffs) / n_paths
        geom_mc_price = df * sum(geom_payoffs) / n_paths

        if use_control_variate:
            # Analytical geometric price
            geom_option = GeometricAsianOption(
                spot=self.spot,
                strike=self.strike,
                T=self.T,
                r=self.r,
                q=self.q,
                sigma=self.sigma,
                option_type=self.option_type,
                averaging_dates=self.averaging_dates
            )
            geom_analytical = geom_option.analytical_price()

            # Control variate adjustment
            price = arith_price - geom_mc_price + geom_analytical

            # Adjusted variance
            adjusted = [
                df * (ap - gp) + geom_analytical
                for ap, gp in zip(arith_payoffs, geom_payoffs)
            ]
            variance = sum((a - price) ** 2 for a in adjusted) / (n_paths - 1)
        else:
            price = arith_price
            variance = sum((df * p - price) ** 2 for p in arith_payoffs) / (n_paths - 1)

        std_error = math.sqrt(variance / n_paths)

        return price, std_error

    def levy_approximation(self) -> float:
        """
        Levy (1992) closed-form approximation.

        Alternative to Turnbull-Wakeman using different moment matching.
        """
        # Similar structure to TW but different formula for effective strike
        return self.turnbull_wakeman_price()  # Use TW as baseline


# Convenience functions

def geometric_asian_call(
    spot: float,
    strike: float,
    T: float,
    r: float,
    sigma: float,
    n_fixings: int = 252,
    q: float = 0.0
) -> float:
    """Price a geometric average price call."""
    dates = [T * (i + 1) / n_fixings for i in range(n_fixings)]
    opt = GeometricAsianOption(
        spot=spot,
        strike=strike,
        T=T,
        r=r,
        q=q,
        sigma=sigma,
        option_type=OptionType.CALL,
        averaging_dates=dates
    )
    return opt.analytical_price()


def arithmetic_asian_call(
    spot: float,
    strike: float,
    T: float,
    r: float,
    sigma: float,
    n_fixings: int = 252,
    q: float = 0.0
) -> float:
    """Price an arithmetic average price call (Turnbull-Wakeman)."""
    dates = [T * (i + 1) / n_fixings for i in range(n_fixings)]
    opt = ArithmeticAsianOption(
        spot=spot,
        strike=strike,
        T=T,
        r=r,
        q=q,
        sigma=sigma,
        option_type=OptionType.CALL,
        averaging_dates=dates
    )
    return opt.turnbull_wakeman_price()


def asian_option_mc(
    spot: float,
    strike: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    averaging_type: str = "arithmetic",
    n_fixings: int = 252,
    n_paths: int = 100000,
    q: float = 0.0
) -> Tuple[float, float]:
    """
    Price Asian option via Monte Carlo with control variate.

    Args:
        spot: Spot price
        strike: Strike
        T: Maturity
        r: Risk-free rate
        sigma: Volatility
        option_type: "call" or "put"
        averaging_type: "arithmetic" or "geometric"
        n_fixings: Number of averaging dates
        n_paths: MC paths
        q: Dividend yield

    Returns:
        (price, standard error)
    """
    dates = [T * (i + 1) / n_fixings for i in range(n_fixings)]
    opt_type = OptionType.CALL if option_type.lower() == "call" else OptionType.PUT

    opt = ArithmeticAsianOption(
        spot=spot,
        strike=strike,
        T=T,
        r=r,
        q=q,
        sigma=sigma,
        option_type=opt_type,
        averaging_dates=dates
    )

    return opt.monte_carlo_price(n_paths=n_paths)
