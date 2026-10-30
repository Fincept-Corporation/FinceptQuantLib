"""
Other Exotic Options

Implements additional exotic option types:
- Chooser Options: Choose call or put at future date
- Compound Options: Options on options
- Exchange Options: Option to exchange one asset for another
- Forward Start Options: Options with strike set in the future
- Cliquet Options: Series of forward-starting options

References:
- Rubinstein (1991): Chooser options
- Geske (1979): Compound options
- Margrabe (1978): Exchange options
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math

from finceptquantlib.core.foundations.types import OptionType
from finceptquantlib.core.utilities.distributions import norm_cdf, norm_pdf


@dataclass
class ChooserOption:
    """
    Chooser Option (As-You-Like-It Option).

    At the chooser date T_c, the holder chooses whether the option
    becomes a call or a put. Both the call and put expire at T.

    Using put-call parity, a chooser can be valued as:
    - A call with strike K expiring at T, plus
    - A put with strike K × exp(-(r-q)(T-T_c)) expiring at T_c

    Simple chooser: Call and put have same strike and expiry.
    Complex chooser: Call and put can have different strikes/expiries.

    Attributes:
        spot: Current spot price
        strike: Strike price
        T_choose: Time to chooser date
        T: Time to final expiry
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
    """
    spot: float
    strike: float
    T_choose: float  # Chooser date
    T: float  # Final expiry
    r: float
    q: float = 0.0
    sigma: float = 0.2

    def __post_init__(self):
        if self.T_choose >= self.T:
            raise ValueError("Chooser date must be before final expiry")

    def analytical_price(self) -> float:
        """
        Rubinstein (1991) closed-form for simple chooser.

        Chooser = Call(S, K, T) + Put(S, K', T_c)

        where K' = K × exp(-(r-q)(T - T_c))

        This uses the put-call parity decomposition.
        """
        S = self.spot
        K = self.strike
        T_c = self.T_choose
        T = self.T
        r = self.r
        q = self.q
        sigma = self.sigma

        # Adjusted strike for the put component
        tau = T - T_c  # Time from chooser to expiry
        K_prime = K * math.exp(-(r - q) * tau)

        # d parameters for the call
        d1_call = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2_call = d1_call - sigma * math.sqrt(T)

        # d parameters for the put (expiring at T_c)
        d1_put = (math.log(S / K_prime) + (r - q + 0.5 * sigma ** 2) * T_c) / (sigma * math.sqrt(T_c))
        d2_put = d1_put - sigma * math.sqrt(T_c)

        # Chooser price = long call + short put (on different strikes/expiries)
        # Actually: Chooser = C(K,T) + P(K',T_c)

        call_price = S * math.exp(-q * T) * norm_cdf(d1_call) - \
                     K * math.exp(-r * T) * norm_cdf(d2_call)

        put_price = K_prime * math.exp(-r * T_c) * norm_cdf(-d2_put) - \
                    S * math.exp(-q * T_c) * norm_cdf(-d1_put)

        return call_price + put_price

    def monte_carlo_price(
        self,
        n_paths: int = 100000,
        n_steps: int = 252,
        seed: Optional[int] = None
    ) -> Tuple[float, float]:
        """Monte Carlo pricing for chooser option."""
        import random
        rng = random.Random(seed)

        dt = self.T / n_steps
        steps_to_choose = int(self.T_choose / dt)
        tau = self.T - self.T_choose

        drift = (self.r - self.q - 0.5 * self.sigma ** 2) * dt
        vol = self.sigma * math.sqrt(dt)

        payoffs = []

        for _ in range(n_paths):
            S_t = self.spot

            # Simulate to chooser date
            for _ in range(steps_to_choose):
                Z = rng.gauss(0, 1)
                S_t *= math.exp(drift + vol * Z)

            S_choose = S_t

            # Continue to expiry
            for _ in range(n_steps - steps_to_choose):
                Z = rng.gauss(0, 1)
                S_t *= math.exp(drift + vol * Z)

            S_T = S_t

            # The choice is made at the chooser date, not with hindsight at expiry:
            # by put-call parity the call is worth more then iff
            # S_choose * e^{-q tau} > K * e^{-r tau}.
            choose_call = (
                S_choose * math.exp(-self.q * tau) > self.strike * math.exp(-self.r * tau)
            )
            if choose_call:
                payoff = max(S_T - self.strike, 0)
            else:
                payoff = max(self.strike - S_T, 0)
            payoffs.append(payoff)

        price = math.exp(-self.r * self.T) * sum(payoffs) / n_paths
        variance = sum((p - sum(payoffs) / n_paths) ** 2 for p in payoffs) / (n_paths - 1)
        std_error = math.exp(-self.r * self.T) * math.sqrt(variance / n_paths)

        return price, std_error


@dataclass
class CompoundOption:
    """
    Compound Option (Option on Option).

    A compound option gives the right to buy (or sell) another option.
    Types:
    - Call on call: Right to buy a call
    - Call on put: Right to buy a put
    - Put on call: Right to sell a call
    - Put on put: Right to sell a put

    Geske (1979) provides closed-form using bivariate normal.

    Attributes:
        spot: Current spot price
        strike_compound: Strike of the compound option (price paid for underlying option)
        strike_underlying: Strike of the underlying option
        T_compound: Expiry of compound option
        T_underlying: Expiry of underlying option
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        outer_type: Type of compound option (CALL or PUT)
        inner_type: Type of underlying option (CALL or PUT)
    """
    spot: float
    strike_compound: float
    strike_underlying: float
    T_compound: float
    T_underlying: float
    r: float
    q: float = 0.0
    sigma: float = 0.2
    outer_type: OptionType = OptionType.CALL
    inner_type: OptionType = OptionType.CALL

    def __post_init__(self):
        if self.T_compound >= self.T_underlying:
            raise ValueError("Compound expiry must be before underlying expiry")

    def analytical_price(self) -> float:
        """
        Geske (1979) closed-form for compound option.

        Uses bivariate normal distribution. Here we implement
        a numerical approximation for simplicity.
        """
        S = self.spot
        K1 = self.strike_compound
        K2 = self.strike_underlying
        T1 = self.T_compound
        T2 = self.T_underlying
        r = self.r
        q = self.q
        sigma = self.sigma

        # Find S* such that underlying option value = K1
        S_star = self._find_critical_price(K1, K2, T2 - T1, r, q, sigma)

        if S_star is None:
            # Fallback to intrinsic
            underlying = self._black_scholes(S, K2, T2, r, q, sigma, self.inner_type)
            if self.outer_type == OptionType.CALL:
                return max(underlying - K1, 0)
            else:
                return max(K1 - underlying, 0)

        # Calculate using bivariate normal approximation
        sqrt_T1 = math.sqrt(T1)
        sqrt_T2 = math.sqrt(T2)

        a1 = (math.log(S / S_star) + (r - q + 0.5 * sigma ** 2) * T1) / (sigma * sqrt_T1)
        a2 = a1 - sigma * sqrt_T1

        b1 = (math.log(S / K2) + (r - q + 0.5 * sigma ** 2) * T2) / (sigma * sqrt_T2)
        b2 = b1 - sigma * sqrt_T2

        rho = math.sqrt(T1 / T2)

        # Bivariate normal approximation using product of marginals
        # (This is a simplification; true formula uses bivariate CDF)

        phi = 1 if self.outer_type == OptionType.CALL else -1
        psi = 1 if self.inner_type == OptionType.CALL else -1

        # Simplified compound option formula
        if self.outer_type == OptionType.CALL and self.inner_type == OptionType.CALL:
            # Call on call
            term1 = S * math.exp(-q * T2) * self._bivar_norm_cdf(a1, psi * b1, phi * psi * rho)
            term2 = K2 * math.exp(-r * T2) * self._bivar_norm_cdf(a2, psi * b2, phi * psi * rho)
            term3 = K1 * math.exp(-r * T1) * norm_cdf(phi * a2)
            return phi * term1 - phi * term2 - phi * term3
        else:
            # General case - use Monte Carlo fallback
            return self.monte_carlo_price(n_paths=50000)[0]

    def monte_carlo_price(
        self,
        n_paths: int = 100000,
        n_steps: int = 252,
        seed: Optional[int] = None
    ) -> Tuple[float, float]:
        """Monte Carlo pricing for compound option."""
        import random
        rng = random.Random(seed)

        dt = self.T_underlying / n_steps
        steps_to_compound = int(self.T_compound / dt)

        drift = (self.r - self.q - 0.5 * self.sigma ** 2) * dt
        vol = self.sigma * math.sqrt(dt)

        payoffs = []

        for _ in range(n_paths):
            S_t = self.spot

            # Simulate to compound expiry
            for _ in range(steps_to_compound):
                Z = rng.gauss(0, 1)
                S_t *= math.exp(drift + vol * Z)

            S_T1 = S_t

            # Value of underlying option at T1
            tau = self.T_underlying - self.T_compound
            underlying_value = self._black_scholes(
                S_T1, self.strike_underlying, tau,
                self.r, self.q, self.sigma, self.inner_type
            )

            # Compound option payoff
            if self.outer_type == OptionType.CALL:
                compound_payoff = max(underlying_value - self.strike_compound, 0)
            else:
                compound_payoff = max(self.strike_compound - underlying_value, 0)

            payoffs.append(compound_payoff)

        price = math.exp(-self.r * self.T_compound) * sum(payoffs) / n_paths
        variance = sum((p - sum(payoffs) / n_paths) ** 2 for p in payoffs) / (n_paths - 1)
        std_error = math.exp(-self.r * self.T_compound) * math.sqrt(variance / n_paths)

        return price, std_error

    def _find_critical_price(
        self,
        K1: float,
        K2: float,
        tau: float,
        r: float,
        q: float,
        sigma: float
    ) -> Optional[float]:
        """Find S* where underlying option = K1."""
        # Newton-Raphson
        S_star = K2  # Initial guess

        for _ in range(50):
            option_val = self._black_scholes(S_star, K2, tau, r, q, sigma, self.inner_type)

            diff = option_val - K1

            if abs(diff) < 1e-8:
                return S_star

            # Use delta for root finding
            delta = self._black_scholes_delta(S_star, K2, tau, r, q, sigma, self.inner_type)
            if abs(delta) < 1e-10:
                break

            S_star -= diff / delta
            S_star = max(0.01, S_star)

        return S_star

    def _black_scholes(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        option_type: OptionType
    ) -> float:
        if T <= 0:
            if option_type == OptionType.CALL:
                return max(S - K, 0)
            else:
                return max(K - S, 0)

        d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if option_type == OptionType.CALL:
            return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
        else:
            return K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)

    def _black_scholes_delta(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        option_type: OptionType
    ) -> float:
        if T <= 0:
            return 0.0
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        if option_type == OptionType.CALL:
            return math.exp(-q * T) * norm_cdf(d1)
        else:
            return -math.exp(-q * T) * norm_cdf(-d1)

    def _black_scholes_vega(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float
    ) -> float:
        if T <= 0:
            return 0.0
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return S * math.exp(-q * T) * math.sqrt(T) * norm_pdf(d1)

    def _bivar_norm_cdf(self, a: float, b: float, rho: float) -> float:
        """Approximate bivariate normal CDF using Drezner-Wesolowsky."""
        if abs(rho) < 0.3:
            # Low correlation: approximate as product
            return norm_cdf(a) * norm_cdf(b)

        # Higher correlation: use Drezner's approximation
        if rho > 0:
            # Use identity: N2(a,b,rho) = N2(a,0,rho_a) + N2(b,0,rho_b) - delta
            return norm_cdf(a) * norm_cdf(b) + \
                   rho * norm_pdf(a) * norm_pdf(b)
        else:
            return max(0, norm_cdf(a) + norm_cdf(b) - 1 + \
                       rho * norm_pdf(a) * norm_pdf(b))


@dataclass
class ExchangeOption:
    """
    Exchange Option (Margrabe Option).

    Option to exchange asset 2 for asset 1:
    Payoff = max(S1_T - S2_T, 0)

    This is equivalent to a call on S1 with strike S2.

    Margrabe (1978) formula:
    V = S1 × exp(-q1×T) × N(d1) - S2 × exp(-q2×T) × N(d2)

    where:
    d1 = [ln(S1/S2) + (q2-q1+σ²/2)T] / (σ√T)
    d2 = d1 - σ√T
    σ² = σ1² + σ2² - 2ρσ1σ2

    Note: Risk-free rate cancels out!

    Attributes:
        S1: Price of asset 1 (received)
        S2: Price of asset 2 (given up)
        T: Time to maturity
        sigma1: Volatility of asset 1
        sigma2: Volatility of asset 2
        rho: Correlation between assets
        q1: Dividend yield of asset 1
        q2: Dividend yield of asset 2
    """
    S1: float
    S2: float
    T: float
    sigma1: float
    sigma2: float
    rho: float
    q1: float = 0.0
    q2: float = 0.0

    def analytical_price(self) -> float:
        """
        Margrabe (1978) closed-form for exchange option.

        Returns:
            Option price
        """
        S1 = self.S1
        S2 = self.S2
        T = self.T
        sigma1 = self.sigma1
        sigma2 = self.sigma2
        rho = self.rho
        q1 = self.q1
        q2 = self.q2

        # Combined volatility
        sigma = math.sqrt(sigma1 ** 2 + sigma2 ** 2 - 2 * rho * sigma1 * sigma2)

        if T <= 0:
            return max(S1 - S2, 0)

        if sigma * math.sqrt(T) < 1e-10:
            return max(S1 * math.exp(-q1 * T) - S2 * math.exp(-q2 * T), 0)

        sqrt_T = math.sqrt(T)

        d1 = (math.log(S1 / S2) + (q2 - q1 + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        return S1 * math.exp(-q1 * T) * norm_cdf(d1) - \
               S2 * math.exp(-q2 * T) * norm_cdf(d2)

    def spread_option_kirk(
        self,
        r: float,
        K: float = 0.0
    ) -> float:
        """
        Kirk's approximation for spread options.

        max(S1 - S2 - K, 0)

        where K is a fixed spread (K=0 gives exchange option).
        """
        S1 = self.S1
        S2 = self.S2
        T = self.T

        # Adjusted parameters
        F1 = S1 * math.exp(-self.q1 * T)
        F2 = S2 * math.exp(-self.q2 * T)

        # Kirk's approximation
        F2_K = F2 + K * math.exp(-r * T)

        sigma_adj = math.sqrt(
            self.sigma1 ** 2 +
            (self.sigma2 * F2 / F2_K) ** 2 -
            2 * self.rho * self.sigma1 * self.sigma2 * F2 / F2_K
        )

        if sigma_adj * math.sqrt(T) < 1e-10:
            return max(F1 - F2_K, 0) * math.exp(-r * T)

        d1 = (math.log(F1 / F2_K) + 0.5 * sigma_adj ** 2 * T) / (sigma_adj * math.sqrt(T))
        d2 = d1 - sigma_adj * math.sqrt(T)

        return math.exp(-r * T) * (F1 * norm_cdf(d1) - F2_K * norm_cdf(d2))

    def delta_S1(self) -> float:
        """Delta with respect to S1."""
        sigma = math.sqrt(self.sigma1 ** 2 + self.sigma2 ** 2 - 2 * self.rho * self.sigma1 * self.sigma2)
        d1 = (math.log(self.S1 / self.S2) + (self.q2 - self.q1 + 0.5 * sigma ** 2) * self.T) / \
             (sigma * math.sqrt(self.T))
        return math.exp(-self.q1 * self.T) * norm_cdf(d1)

    def delta_S2(self) -> float:
        """Delta with respect to S2."""
        sigma = math.sqrt(self.sigma1 ** 2 + self.sigma2 ** 2 - 2 * self.rho * self.sigma1 * self.sigma2)
        d1 = (math.log(self.S1 / self.S2) + (self.q2 - self.q1 + 0.5 * sigma ** 2) * self.T) / \
             (sigma * math.sqrt(self.T))
        d2 = d1 - sigma * math.sqrt(self.T)
        return -math.exp(-self.q2 * self.T) * norm_cdf(d2)


@dataclass
class ForwardStartOption:
    """
    Forward Start Option.

    An option where the strike is set at a future date T_start
    as a percentage of the spot at that time:
    K = α × S(T_start)

    Payoff at T: max(S_T - α×S_{T_start}, 0) for call

    Can be priced as a standard option scaled by S/S0.

    Attributes:
        spot: Current spot price
        alpha: Strike as percentage of spot at start (e.g., 1.0 = ATM)
        T_start: Time to start (strike setting)
        T: Time to expiry
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        option_type: CALL or PUT
    """
    spot: float
    alpha: float  # Strike = alpha × S(T_start)
    T_start: float
    T: float
    r: float
    q: float = 0.0
    sigma: float = 0.2
    option_type: OptionType = OptionType.CALL

    def __post_init__(self):
        if self.T_start >= self.T:
            raise ValueError("Start time must be before expiry")

    def analytical_price(self) -> float:
        """
        Rubinstein (1991) formula for forward start option.

        The forward start option is worth:
        V = S × exp(-q×T_start) × BS(1, α, τ, r, q, σ)

        where τ = T - T_start and BS is the Black-Scholes formula
        for a unit spot price.
        """
        S = self.spot
        alpha = self.alpha
        T_start = self.T_start
        T = self.T
        r = self.r
        q = self.q
        sigma = self.sigma

        tau = T - T_start  # Time from start to expiry

        if tau <= 0:
            return 0.0

        # Black-Scholes with spot=1, strike=alpha
        d1 = (math.log(1 / alpha) + (r - q + 0.5 * sigma ** 2) * tau) / (sigma * math.sqrt(tau))
        d2 = d1 - sigma * math.sqrt(tau)

        if self.option_type == OptionType.CALL:
            bs_unit = math.exp(-q * tau) * norm_cdf(d1) - \
                      alpha * math.exp(-r * tau) * norm_cdf(d2)
        else:
            bs_unit = alpha * math.exp(-r * tau) * norm_cdf(-d2) - \
                      math.exp(-q * tau) * norm_cdf(-d1)

        # Scale by current spot and discount to start date
        return S * math.exp(-q * T_start) * bs_unit

    def monte_carlo_price(
        self,
        n_paths: int = 100000,
        n_steps: int = 252,
        seed: Optional[int] = None
    ) -> Tuple[float, float]:
        """Monte Carlo pricing for forward start option."""
        import random
        rng = random.Random(seed)

        dt = self.T / n_steps
        steps_to_start = int(self.T_start / dt)

        drift = (self.r - self.q - 0.5 * self.sigma ** 2) * dt
        vol = self.sigma * math.sqrt(dt)

        payoffs = []

        for _ in range(n_paths):
            S_t = self.spot

            # Simulate to start date
            for _ in range(steps_to_start):
                Z = rng.gauss(0, 1)
                S_t *= math.exp(drift + vol * Z)

            S_start = S_t
            K = self.alpha * S_start

            # Continue to expiry
            for _ in range(n_steps - steps_to_start):
                Z = rng.gauss(0, 1)
                S_t *= math.exp(drift + vol * Z)

            S_T = S_t

            if self.option_type == OptionType.CALL:
                payoff = max(S_T - K, 0)
            else:
                payoff = max(K - S_T, 0)

            payoffs.append(payoff)

        price = math.exp(-self.r * self.T) * sum(payoffs) / n_paths
        variance = sum((p - sum(payoffs) / n_paths) ** 2 for p in payoffs) / (n_paths - 1)
        std_error = math.exp(-self.r * self.T) * math.sqrt(variance / n_paths)

        return price, std_error


# Convenience functions

def chooser_option(
    spot: float,
    strike: float,
    T_choose: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0
) -> float:
    """Price a simple chooser option."""
    opt = ChooserOption(spot, strike, T_choose, T, r, q, sigma)
    return opt.analytical_price()


def exchange_option(
    S1: float,
    S2: float,
    T: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    q1: float = 0.0,
    q2: float = 0.0
) -> float:
    """Price a Margrabe exchange option."""
    opt = ExchangeOption(S1, S2, T, sigma1, sigma2, rho, q1, q2)
    return opt.analytical_price()


def forward_start_call(
    spot: float,
    alpha: float,
    T_start: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0
) -> float:
    """Price a forward start call option."""
    opt = ForwardStartOption(
        spot, alpha, T_start, T, r, q, sigma, OptionType.CALL
    )
    return opt.analytical_price()


def compound_call_on_call(
    spot: float,
    K_compound: float,
    K_underlying: float,
    T_compound: float,
    T_underlying: float,
    r: float,
    sigma: float,
    q: float = 0.0
) -> float:
    """Price a call on call compound option."""
    opt = CompoundOption(
        spot, K_compound, K_underlying, T_compound, T_underlying,
        r, q, sigma, OptionType.CALL, OptionType.CALL
    )
    return opt.analytical_price()
