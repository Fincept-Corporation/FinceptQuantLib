"""
Tests for automatic differentiation engine.

Validates correctness of AD implementation against known analytical derivatives.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from finceptquantlib.core.foundations.autodiff import Dual, HyperDual, gradient, hessian, jacobian, taylor_expand
from finceptquantlib.core.utilities.math_utils import exp, log, sin, cos, power


class TestDual:
    """Test Dual number arithmetic and calculus."""

    def test_basic_construction(self) -> None:
        """Test Dual number creation."""
        x = Dual(2.0, 1.0)
        assert x.value == 2.0
        assert x.deriv == 1.0

    def test_addition(self) -> None:
        """Test (f + g)' = f' + g'."""
        x = Dual(2.0, 1.0)
        y = Dual(3.0, 0.5)
        z = x + y

        assert z.value == 5.0
        assert z.deriv == 1.5

        # Scalar addition
        w = x + 10
        assert w.value == 12.0
        assert w.deriv == 1.0

    def test_subtraction(self) -> None:
        """Test (f - g)' = f' - g'."""
        x = Dual(5.0, 1.0)
        y = Dual(2.0, 0.5)
        z = x - y

        assert z.value == 3.0
        assert z.deriv == 0.5

    def test_multiplication(self) -> None:
        """Test (fg)' = f'g + fg'."""
        x = Dual(3.0, 1.0)
        y = Dual(2.0, 0.5)
        z = x * y

        assert z.value == 6.0
        assert z.deriv == 3.0 * 0.5 + 2.0 * 1.0  # 1.5 + 2.0 = 3.5

    def test_division(self) -> None:
        """Test (f/g)' = (f'g - fg')/g²."""
        x = Dual(6.0, 1.0)
        y = Dual(2.0, 0.5)
        z = x / y

        assert z.value == 3.0
        # (1.0 * 2.0 - 6.0 * 0.5) / 4.0 = (2.0 - 3.0) / 4.0 = -0.25
        assert z.deriv == -0.25

    def test_power(self) -> None:
        """Test (x^n)' = n*x^(n-1) * x'."""
        x = Dual(2.0, 1.0)
        y = x ** 3

        assert y.value == 8.0
        assert y.deriv == 3 * 4 * 1.0  # 3 * 2^2 = 12

    def test_polynomial(self) -> None:
        """Test polynomial: f(x) = x^2 + 3x + 1, f'(x) = 2x + 3."""
        x = Dual(2.0, 1.0)
        y = x**2 + 3*x + 1

        assert y.value == 11.0  # 4 + 6 + 1
        assert y.deriv == 7.0   # 2*2 + 3

    def test_chain_rule(self) -> None:
        """Test chain rule with nested functions."""
        x = Dual(1.0, 1.0)
        # f(x) = (x^2 + 1)^3
        # f'(x) = 3(x^2 + 1)^2 * 2x = 6x(x^2 + 1)^2
        y = (x**2 + 1)**3

        assert y.value == 8.0  # (1 + 1)^3
        assert y.deriv == 6 * 1 * 4  # 6 * 1 * (2)^2 = 24

    def test_exponential(self) -> None:
        """Test e^x has derivative e^x."""
        x = Dual(1.0, 1.0)
        y = exp(x)

        assert_allclose(y.value, np.e)
        assert_allclose(y.deriv, np.e)

    def test_logarithm(self) -> None:
        """Test ln(x) has derivative 1/x."""
        x = Dual(2.0, 1.0)
        y = log(x)

        assert_allclose(y.value, np.log(2.0))
        assert_allclose(y.deriv, 0.5)  # 1/2

    def test_sine(self) -> None:
        """Test sin'(x) = cos(x)."""
        x = Dual(np.pi/4, 1.0)
        y = sin(x)

        assert_allclose(y.value, np.sqrt(2)/2)
        assert_allclose(y.deriv, np.sqrt(2)/2)  # cos(π/4)

    def test_cosine(self) -> None:
        """Test cos'(x) = -sin(x)."""
        x = Dual(np.pi/4, 1.0)
        y = cos(x)

        assert_allclose(y.value, np.sqrt(2)/2)
        assert_allclose(y.deriv, -np.sqrt(2)/2)  # -sin(π/4)


class TestHyperDual:
    """Test HyperDual numbers for higher-order derivatives."""

    def test_second_derivative_polynomial(self) -> None:
        """Test f(x) = x^3, f''(x) = 6x."""
        x = HyperDual(2.0, 1.0, 0.0)
        y = x ** 3

        assert y.value == 8.0
        assert y.d1 == 12.0  # 3 * 4
        assert y.d2 == 12.0  # 6 * 2

    def test_third_derivative(self) -> None:
        """Test f(x) = x^4, f'''(x) = 24x."""
        x = HyperDual(2.0, 1.0, 0.0, 0.0)
        y = x ** 4

        assert y.value == 16.0
        assert y.d1 == 32.0   # 4 * 8
        assert y.d2 == 48.0   # 12 * 4
        assert y.d3 == 48.0   # 24 * 2

    def test_fourth_derivative(self) -> None:
        """Test f(x) = x^5, f''''(x) = 120x."""
        x = HyperDual(2.0, 1.0, 0.0, 0.0, 0.0)
        y = x ** 5

        assert y.value == 32.0
        assert y.d1 == 80.0    # 5 * 16
        assert y.d2 == 160.0   # 20 * 8
        assert y.d3 == 240.0   # 60 * 4
        assert y.d4 == 240.0   # 120 * 2

    def test_exponential_all_derivatives_equal(self) -> None:
        """Test e^x has all derivatives equal to e^x."""
        x = HyperDual(1.0, 1.0, 0.0, 0.0, 0.0)
        y = exp(x)

        assert_allclose(y.value, np.e)
        assert_allclose(y.d1, np.e)
        assert_allclose(y.d2, np.e)
        assert_allclose(y.d3, np.e)
        assert_allclose(y.d4, np.e)

    def test_logarithm_higher_derivatives(self) -> None:
        """
        Test ln(x) derivatives:
        f'(x) = 1/x
        f''(x) = -1/x²
        f'''(x) = 2/x³
        f''''(x) = -6/x⁴
        """
        x = HyperDual(2.0, 1.0, 0.0, 0.0, 0.0)
        y = log(x)

        assert_allclose(y.value, np.log(2.0))
        assert_allclose(y.d1, 0.5)         # 1/2
        assert_allclose(y.d2, -0.25)       # -1/4
        assert_allclose(y.d3, 0.25)        # 2/8
        assert_allclose(y.d4, -0.375)      # -6/16

    def test_cross_term_product_matches_power(self) -> None:
        """x·x and x**2 must carry the same ε₁ε₂ cross term (2 for a unit seed)."""
        x = HyperDual(3.0, 1.0)
        assert (x * x).d12 == pytest.approx((x ** 2).d12)
        assert (x * Dual(3.0, 1.0)).d12 == pytest.approx(2.0)


class TestGradient:
    """Test gradient computation."""

    def test_gradient_scalar(self) -> None:
        """Test gradient of f(x) = x^2."""
        def f(x):
            return x ** 2

        grad = gradient(f, 3.0)
        assert_allclose(grad, 6.0)  # f'(3) = 2*3

    def test_gradient_vector(self) -> None:
        """Test gradient of f(x,y) = x^2 + y^2."""
        def f(x):
            return x[0]**2 + x[1]**2

        grad = gradient(f, np.array([1.0, 2.0]))
        assert_allclose(grad, np.array([2.0, 4.0]))

    def test_gradient_multivariate(self) -> None:
        """Test ∇f where f(x,y,z) = x^2 + 2xy + z^3."""
        def f(x):
            return x[0]**2 + 2*x[0]*x[1] + x[2]**3

        grad = gradient(f, np.array([1.0, 2.0, 1.0]))
        # ∂f/∂x = 2x + 2y = 2 + 4 = 6
        # ∂f/∂y = 2x = 2
        # ∂f/∂z = 3z^2 = 3
        assert_allclose(grad, np.array([6.0, 2.0, 3.0]))


class TestHessian:
    """Test Hessian matrix computation."""

    def test_hessian_scalar(self) -> None:
        """Test Hessian of f(x) = x^3."""
        def f(x):
            return x ** 3

        H = hessian(f, 2.0)
        assert_allclose(H, 12.0)  # f''(2) = 6*2

    def test_hessian_quadratic(self) -> None:
        """Test Hessian of f(x,y) = x^2 + xy + y^2."""
        def f(x):
            return x[0]**2 + x[0]*x[1] + x[1]**2

        H = hessian(f, np.array([1.0, 1.0]))
        # ∂²f/∂x² = 2, ∂²f/∂y² = 2, ∂²f/∂x∂y = 1
        expected = np.array([[2.0, 1.0],
                             [1.0, 2.0]])
        assert_allclose(H, expected)

    def test_hessian_mixed_partials_transcendental(self) -> None:
        """Test Hessian of f(x,y) = exp(xy) + x·sin(y) against analytic partials."""
        def f(v):
            return exp(v[0] * v[1]) + v[0] * sin(v[1])

        x, y = 0.5, 1.2
        e = np.exp(x * y)
        expected = np.array([
            [y * y * e, e * (1 + x * y) + np.cos(y)],
            [e * (1 + x * y) + np.cos(y), x * x * e - x * np.sin(y)],
        ])
        assert_allclose(hessian(f, [x, y]), expected, rtol=1e-12)


class TestJacobian:
    """Test Jacobian matrix computation."""

    def test_jacobian_vector_function(self) -> None:
        """Test Jacobian of F(x,y) = [x^2, y^2, xy]."""
        def f(x):
            return np.array([x[0]**2, x[1]**2, x[0]*x[1]])

        J = jacobian(f, np.array([1.0, 2.0]))
        # ∂f₁/∂x = 2x = 2, ∂f₁/∂y = 0
        # ∂f₂/∂x = 0, ∂f₂/∂y = 2y = 4
        # ∂f₃/∂x = y = 2, ∂f₃/∂y = x = 1
        expected = np.array([[2.0, 0.0],
                             [0.0, 4.0],
                             [2.0, 1.0]])
        assert_allclose(J, expected)


class TestTaylorSeries:
    """Test Taylor series expansion."""

    def test_taylor_polynomial(self) -> None:
        """Test Taylor expansion of x^3 around x=1."""
        def f(x):
            return x ** 3

        coeffs = taylor_expand(f, 1.0, order=3)
        # f(1) = 1, f'(1) = 3, f''(1)/2! = 3, f'''(1)/3! = 1
        assert_allclose(coeffs, np.array([1.0, 3.0, 3.0, 1.0]))

    def test_taylor_exponential(self) -> None:
        """Test Taylor expansion of e^x around x=0."""
        def f(x):
            return exp(x)

        coeffs = taylor_expand(f, 0.0, order=4)
        # All derivatives of e^x = 1 at x=0
        assert_allclose(coeffs, np.array([1.0, 1.0, 0.5, 1.0/6, 1.0/24]), rtol=1e-10)


class TestComplexExpressions:
    """Test AD on complex mathematical expressions."""

    def test_black_scholes_formula(self) -> None:
        """
        Test Black-Scholes call option delta.

        C(S) = S*N(d1) - K*exp(-rT)*N(d2)
        Delta = ∂C/∂S = N(d1) + S*N'(d1)*∂d1/∂S - K*exp(-rT)*N'(d2)*∂d2/∂S

        For simplicity, test delta of simplified payoff: max(S-K, 0)
        """
        def call_payoff(S):
            K = 100.0
            return power(S - K, 1.0) if S >= K else Dual(0.0, 0.0)

        S = Dual(105.0, 1.0)
        payoff = call_payoff(S)

        assert payoff.value == 5.0
        assert payoff.deriv == 1.0  # Delta = 1 when ITM

    def test_bond_duration(self) -> None:
        """
        Test modified duration calculation using AD.

        Price = ∑ CF_i / (1+y)^t_i
        Modified Duration = -1/P * dP/dy
        """
        def bond_price(y):
            # Simple 3-year annual bond, 5% coupon, par = 100
            cash_flows = [5.0, 5.0, 105.0]
            times = [1.0, 2.0, 3.0]

            price = sum(cf / (1 + y)**t for cf, t in zip(cash_flows, times))
            return price

        y = Dual(0.05, 1.0)  # 5% yield
        P = bond_price(y)

        # At par, modified duration ≈ Macaulay duration ≈ 2.86 years
        mod_duration = -P.deriv / P.value
        assert 2.7 < mod_duration < 3.0  # Approximate check


class TestHigherDimensional:
    """Gradients and Hessians of multi-dimensional functions."""

    def test_rosenbrock_gradient(self) -> None:
        def rosenbrock(x):
            return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
                       for i in range(len(x) - 1))

        x = np.random.default_rng(0).standard_normal(10).tolist()
        result = gradient(rosenbrock, x)

        expected = [0.0] * 10
        for i in range(9):
            expected[i] += -400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i])
            expected[i+1] += 200 * (x[i+1] - x[i]**2)
        assert result == pytest.approx(expected, rel=1e-12, abs=1e-12)

    def test_quadratic_hessian(self) -> None:
        def quadratic(x):
            return sum((i + 1) * x[i]**2 for i in range(len(x)))

        x = np.random.default_rng(1).standard_normal(5).tolist()
        result = hessian(quadratic, x)

        for i in range(5):
            for j in range(5):
                assert result[i][j] == pytest.approx(2.0 * (i + 1) if i == j else 0.0, abs=1e-10)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
