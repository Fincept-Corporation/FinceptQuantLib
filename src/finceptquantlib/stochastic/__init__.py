"""
Stochastic Calculus Module

Comprehensive stochastic calculus toolkit:

- processes/: Stochastic processes (Brownian, Poisson, Lévy, jump-diffusion)
- simulation/: SDE discretization schemes (Euler-Maruyama, Milstein, exact, MLMC)
- theory/: Theoretical tools (Ito's lemma, Girsanov, martingales)

Example:
    >>> from finceptquantlib.stochastic import GeometricBrownianMotion, euler_maruyama
    >>>
    >>> # Simulate GBM
    >>> gbm = GeometricBrownianMotion(mu=0.05, sigma=0.2, S0=100)
    >>> times, paths = gbm.simulate(T=1.0, n_steps=252, n_paths=1000)
    >>>
    >>> # Generic SDE with Euler-Maruyama
    >>> drift = lambda x, t: 0.05 * x
    >>> diffusion = lambda x, t: 0.2 * x
    >>> result = euler_maruyama(drift, diffusion, X0=100, T=1.0, n_steps=252)
"""

# Processes
from finceptquantlib.stochastic.processes import (
    # Brownian
    WienerProcess,
    GeometricBrownianMotion,
    BrownianBridge,
    correlated_brownian,
    reflected_brownian,
    # Mean-reverting
    OrnsteinUhlenbeck,
    CoxIngersollRoss,
    Vasicek,
    HullWhite,
    # Poisson
    PoissonProcess,
    InhomogeneousPoissonProcess,
    CompoundPoissonProcess,
    normal_jump_sampler,
    double_exponential_sampler,
    uniform_jump_sampler,
    # Jump-diffusion (new names - preferred)
    MertonProcess,
    KouProcess,
    BatesProcess,
    BatesModel,
    # Lévy
    VarianceGammaProcess,
    NormalInverseGaussian,
    CGMYProcess,
)

# Simulation
from finceptquantlib.stochastic.simulation import (
    # Euler-Maruyama
    euler_maruyama,
    euler_maruyama_nd,
    euler_maruyama_with_jumps,
    EulerMaruyamaResult,
    # Milstein
    milstein,
    milstein_implicit,
    milstein_nd,
    MilsteinResult,
    # Exact
    simulate_gbm_exact,
    simulate_ou_exact,
    simulate_cir_exact,
    simulate_heston_exact,
    ExactSimulationResult,
    # MLMC
    mlmc_estimate,
    mlmc_coupled_paths,
    MLMCResult,
)

# Theory
from finceptquantlib.stochastic.theory import (
    # Ito
    ItoProcess,
    ito_lemma_1d,
    ito_product_rule,
    quadratic_variation,
    covariation,
    stochastic_integral_approx,
    ito_to_stratonovich,
    stratonovich_to_ito,
    # Girsanov
    MeasureChange,
    risk_neutral_drift,
    esscher_transform,
    simulate_under_Q,
    change_of_numeraire,
    # Martingale
    is_martingale_test,
    optional_stopping_value,
    doob_maximal_inequality,
    quadratic_variation_martingale,
    predictable_representation,
    local_martingale_test,
)

# Sampling utilities
from finceptquantlib.stochastic.sampling import (
    # Poisson
    poisson_sample,
    poisson_samples,
    # Exponential
    exponential_sample,
    exponential_samples,
    # Gamma
    gamma_sample,
    gamma_samples,
    # Chi-squared
    chi_squared_sample,
    # Beta
    beta_sample,
    # Inverse Gaussian
    inverse_gaussian_sample,
    # Multivariate
    multivariate_normal_sample,
    correlated_normals,
    # Variance reduction
    antithetic_normal_samples,
    sobol_sequence,
)

__all__ = [
    # Processes - Brownian
    "WienerProcess",
    "GeometricBrownianMotion",
    "BrownianBridge",
    "correlated_brownian",
    "reflected_brownian",
    # Processes - Mean-reverting
    "OrnsteinUhlenbeck",
    "CoxIngersollRoss",
    "Vasicek",
    "HullWhite",
    # Processes - Poisson
    "PoissonProcess",
    "InhomogeneousPoissonProcess",
    "CompoundPoissonProcess",
    "normal_jump_sampler",
    "double_exponential_sampler",
    "uniform_jump_sampler",
    # Processes - Jump-diffusion (new names - preferred)
    "MertonProcess",
    "KouProcess",
    "BatesProcess",
    "BatesModel",
    # Processes - Lévy
    "VarianceGammaProcess",
    "NormalInverseGaussian",
    "CGMYProcess",
    # Simulation
    "euler_maruyama",
    "euler_maruyama_nd",
    "euler_maruyama_with_jumps",
    "EulerMaruyamaResult",
    "milstein",
    "milstein_implicit",
    "milstein_nd",
    "MilsteinResult",
    "simulate_gbm_exact",
    "simulate_ou_exact",
    "simulate_cir_exact",
    "simulate_heston_exact",
    "ExactSimulationResult",
    "mlmc_estimate",
    "mlmc_coupled_paths",
    "MLMCResult",
    # Theory - Ito
    "ItoProcess",
    "ito_lemma_1d",
    "ito_product_rule",
    "quadratic_variation",
    "covariation",
    "stochastic_integral_approx",
    "ito_to_stratonovich",
    "stratonovich_to_ito",
    # Theory - Girsanov
    "MeasureChange",
    "risk_neutral_drift",
    "esscher_transform",
    "simulate_under_Q",
    "change_of_numeraire",
    # Theory - Martingale
    "is_martingale_test",
    "optional_stopping_value",
    "doob_maximal_inequality",
    "quadratic_variation_martingale",
    "predictable_representation",
    "local_martingale_test",
    # Sampling
    "poisson_sample",
    "poisson_samples",
    "exponential_sample",
    "exponential_samples",
    "gamma_sample",
    "gamma_samples",
    "chi_squared_sample",
    "beta_sample",
    "inverse_gaussian_sample",
    "multivariate_normal_sample",
    "correlated_normals",
    "antithetic_normal_samples",
    "sobol_sequence",
]
