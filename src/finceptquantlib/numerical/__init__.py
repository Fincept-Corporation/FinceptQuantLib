"""
Numerical Methods Module

Pure numerical algorithms for mathematical computation:
- solvers/: Root finding, optimization, least squares
- interpolation/: Polynomial, spline, monotone interpolation
- integration/: Quadrature, Monte Carlo integration
- differentiation/: Finite differences, automatic differentiation
- linalg/: Linear algebra operations
- ode/: ODE/PDE solvers
"""

# Solvers
from finceptquantlib.numerical.solvers import (
    # 1D Root Finding
    brent,
    newton,
    bisect,
    secant,
    ridder,
    RootResult,
    find_root,
    find_root_newton,
    # Multi-dim Root Finding
    NewtonRaphson,
    Broyden,
    RootResultND,
    find_root_nd,
    # Least Squares
    LevenbergMarquardt,
    BoundedLevenbergMarquardt,
    GaussNewton,
    LeastSquaresResult,
    least_squares,
    # Optimization
    nelder_mead,
    bfgs,
    OptimizeResult,
)

# Linear Algebra
from finceptquantlib.numerical.linalg import (
    solve,
    lstsq,
    lu_decompose,
    cholesky,
    qr_decompose,
    svd,
    dot,
    norm,
    matvec,
    transpose,
    matrix_inverse,
)

# Differentiation
from finceptquantlib.numerical.differentiation import (
    finite_difference_jacobian,
    finite_difference_gradient,
    finite_difference_derivative,
    finite_difference_hessian,
    JacobianComputer,
    parallel_jacobian,
    parallel_gradient,
    ParallelJacobianComputer,
)

# Interpolation
from finceptquantlib.numerical.interpolation import (
    linear_interp,
    cubic_interp,
    lagrange_interp,
    cubic_spline,
    monotone_convex,
    pchip,
)

# Integration
from finceptquantlib.numerical.integration import (
    simpson,
    trapezoid,
    gauss_legendre,
    adaptive_quad,
    monte_carlo_integrate,
    romberg,
    stratified_sampling,
    importance_sampling,
)

# FFT
from finceptquantlib.numerical.fft import (
    cooley_tukey_fft,
    ifft,
    fft_real,
    fft_convolve,
    carr_madan_fft,
    lewis_fft,
    cos_method,
    next_power_of_2,
    pad_to_power_of_2,
)

__all__ = [
    # 1D Root Finding
    "brent",
    "newton",
    "bisect",
    "secant",
    "ridder",
    "RootResult",
    "find_root",
    "find_root_newton",
    # Multi-dim Root Finding
    "NewtonRaphson",
    "Broyden",
    "RootResultND",
    "find_root_nd",
    # Least Squares
    "LevenbergMarquardt",
    "BoundedLevenbergMarquardt",
    "GaussNewton",
    "LeastSquaresResult",
    "least_squares",
    # Optimization
    "nelder_mead",
    "bfgs",
    "OptimizeResult",
    # Linear Algebra
    "solve",
    "lstsq",
    "lu_decompose",
    "cholesky",
    "qr_decompose",
    "svd",
    "dot",
    "norm",
    "matvec",
    "transpose",
    "matrix_inverse",
    # Differentiation
    "finite_difference_jacobian",
    "finite_difference_gradient",
    "finite_difference_derivative",
    "finite_difference_hessian",
    "JacobianComputer",
    "parallel_jacobian",
    "parallel_gradient",
    "ParallelJacobianComputer",
    # Interpolation
    "linear_interp",
    "cubic_interp",
    "lagrange_interp",
    "cubic_spline",
    "monotone_convex",
    "pchip",
    # Integration
    "simpson",
    "trapezoid",
    "gauss_legendre",
    "adaptive_quad",
    "monte_carlo_integrate",
    "romberg",
    "stratified_sampling",
    "importance_sampling",
    # FFT
    "cooley_tukey_fft",
    "ifft",
    "fft_real",
    "fft_convolve",
    "carr_madan_fft",
    "lewis_fft",
    "cos_method",
    "next_power_of_2",
    "pad_to_power_of_2",
]
