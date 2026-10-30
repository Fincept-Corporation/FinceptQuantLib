"""
Comprehensive Test Suite for FinceptQuantLib

Tests all major modules:
- Core (autodiff)
- Scheduling (day counts, calendars)
- Curves (interpolation, parametric)
- Instruments (bonds, swaps, CDS, FX)
- Risk (VaR, stress testing)
"""

import pytest
import numpy as np
from datetime import date
from numpy.testing import assert_almost_equal, assert_array_almost_equal


# ============================================================================
# Core Module Tests
# ============================================================================

class TestAutoDiff:
    """Tests for automatic differentiation."""

    def test_dual_basic_ops(self):
        """Test basic Dual number operations."""
        from finceptquantlib.core import Dual

        x = Dual(3.0, 1.0)
        y = Dual(2.0, 0.0)

        # Addition
        z = x + y
        assert z.value == 5.0
        assert z.deriv == 1.0

        # Multiplication
        z = x * y
        assert z.value == 6.0
        assert z.deriv == 2.0

        # Power
        z = x ** 2
        assert z.value == 9.0
        assert z.deriv == 6.0

    def test_dual_transcendental(self):
        """Test transcendental functions with Dual."""
        from finceptquantlib.core import Dual

        x = Dual(0.0, 1.0)

        # exp
        z = Dual.exp(x)
        assert_almost_equal(z.value, 1.0)
        assert_almost_equal(z.deriv, 1.0)

        # sin
        z = Dual.sin(x)
        assert_almost_equal(z.value, 0.0)
        assert_almost_equal(z.deriv, 1.0)  # cos(0) = 1

    def test_gradient(self):
        """Test gradient computation."""
        from finceptquantlib.core import gradient

        def f(x):
            return x[0]**2 + x[1]**2

        x = np.array([3.0, 4.0])
        grad = gradient(f, x)

        assert_array_almost_equal(grad, [6.0, 8.0])

    def test_hessian(self):
        """Test Hessian computation."""
        from finceptquantlib.core import hessian

        def f(x):
            return x[0]**2 + x[1]**2 + x[0]*x[1]

        x = np.array([1.0, 1.0])
        H = hessian(f, x)

        # Hessian should be [[2, 1], [1, 2]]
        assert_array_almost_equal(H, [[2.0, 1.0], [1.0, 2.0]], decimal=4)


# ============================================================================
# Scheduling Module Tests
# ============================================================================

class TestDayCount:
    """Tests for day count conventions."""

    def test_actual_360(self):
        """Test ACT/360 day count."""
        from finceptquantlib.scheduling import Actual360

        dc = Actual360()
        d1 = date(2024, 1, 1)
        d2 = date(2024, 7, 1)

        # 182 days / 360
        dcf = dc.year_fraction(d1, d2)
        assert_almost_equal(dcf, 182/360, decimal=6)

    def test_actual_365(self):
        """Test ACT/365F day count."""
        from finceptquantlib.scheduling import Actual365Fixed

        dc = Actual365Fixed()
        d1 = date(2024, 1, 1)
        d2 = date(2025, 1, 1)

        # 366 days in leap year / 365
        dcf = dc.year_fraction(d1, d2)
        assert_almost_equal(dcf, 366/365, decimal=6)

    def test_thirty_360(self):
        """Test 30/360 day count."""
        from finceptquantlib.scheduling import Thirty360

        dc = Thirty360()
        d1 = date(2024, 1, 31)
        d2 = date(2024, 3, 31)

        # 60 days / 360 = 1/6
        dcf = dc.year_fraction(d1, d2)
        assert_almost_equal(dcf, 60/360, decimal=6)


class TestCalendar:
    """Tests for business day calendars."""

    def test_target_calendar(self):
        """Test TARGET calendar."""
        from finceptquantlib.scheduling import TargetCalendar

        cal = TargetCalendar()

        # New Year's Day
        assert not cal.is_business_day(date(2024, 1, 1))

        # Regular Monday
        assert cal.is_business_day(date(2024, 1, 8))

        # Weekend
        assert not cal.is_business_day(date(2024, 1, 6))  # Saturday

    def test_us_federal_calendar(self):
        """Test US Federal calendar."""
        from finceptquantlib.scheduling import USFederalCalendar

        cal = USFederalCalendar()

        # Independence Day
        assert not cal.is_business_day(date(2024, 7, 4))

        # Regular business day
        assert cal.is_business_day(date(2024, 7, 5))


class TestSchedule:
    """Tests for payment schedules."""

    def test_quarterly_schedule(self):
        """Test quarterly payment schedule generation."""
        from finceptquantlib.scheduling import Schedule, Frequency

        schedule = Schedule(
            effective_date=date(2024, 1, 1),
            termination_date=date(2025, 1, 1),
            frequency=Frequency.QUARTERLY,
        )

        periods = schedule.periods
        assert len(periods) == 4

        # Check first period
        assert periods[0].start_date == date(2024, 1, 1)
        assert periods[0].end_date == date(2024, 4, 1)


# ============================================================================
# Curves Module Tests
# ============================================================================

class TestCurves:
    """Tests for yield curves."""

    def test_zero_curve(self):
        """Test ZeroCurve interpolation."""
        from finceptquantlib.curves import ZeroCurve

        curve = ZeroCurve(
            reference_date=date(2024, 1, 1),
            tenors=[0.25, 0.5, 1.0, 2.0, 5.0],
            values=[0.05, 0.051, 0.052, 0.053, 0.055],
        )

        # Test interpolation at 3 years
        rate = curve.zero_rate(3.0)
        assert 0.053 < rate < 0.055

        # Test discount factor
        df = curve.discount_factor(1.0)
        expected_df = np.exp(-0.052 * 1.0)
        assert_almost_equal(df, expected_df, decimal=4)

    def test_discount_curve(self):
        """Test DiscountCurve."""
        from finceptquantlib.curves import DiscountCurve

        curve = DiscountCurve(
            reference_date=date(2024, 1, 1),
            tenors=[0.25, 0.5, 1.0, 2.0],
            values=[0.99, 0.98, 0.95, 0.90],
        )

        # DF at t=0 should be 1
        assert_almost_equal(curve.discount_factor(0), 1.0)

        # Test forward rate
        fwd = curve.forward_rate(0.5, 1.0)
        assert fwd > 0

    def test_nelson_siegel(self):
        """Test Nelson-Siegel parametric curve."""
        from finceptquantlib.curves.parametric import NelsonSiegelCurve

        # Typical parameters
        curve = NelsonSiegelCurve(
            reference_date=date(2024, 1, 1),
            beta0=0.05,   # Long-term level
            beta1=-0.02,  # Slope
            beta2=0.01,   # Curvature
            tau=2.0,
        )

        # Long-term rate should approach beta0
        rate_30y = curve.zero_rate(30.0)
        assert_almost_equal(rate_30y, 0.05, decimal=2)

        # Short rate should be beta0 + beta1
        rate_short = curve.zero_rate(0.01)
        assert_almost_equal(rate_short, 0.03, decimal=1)


# ============================================================================
# Instruments Module Tests
# ============================================================================

class TestBonds:
    """Tests for bond pricing."""

    def test_fixed_rate_bond(self):
        """Test fixed rate bond pricing."""
        from finceptquantlib.instruments import FixedRateBond
        from finceptquantlib.curves import ZeroCurve

        curve = ZeroCurve(
            reference_date=date(2024, 1, 1),
            tenors=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
            values=[0.05] * 6,
        )

        bond = FixedRateBond(
            notional=100,
            coupon_rate=0.05,
            issue_date=date(2024, 1, 1),
            maturity_date=date(2029, 1, 1),
            frequency="semi-annual",
        )

        result = bond.price(curve, date(2024, 1, 1))

        # At par yield, price should be close to 100
        assert 99 < result.clean_price < 101

    def test_zero_coupon_bond(self):
        """Test zero coupon bond."""
        from finceptquantlib.instruments import ZeroCouponBond
        from finceptquantlib.curves import ZeroCurve

        curve = ZeroCurve(
            reference_date=date(2024, 1, 1),
            tenors=[1.0, 2.0, 5.0],
            values=[0.05, 0.05, 0.05],
        )

        bond = ZeroCouponBond(
            notional=100,
            issue_date=date(2024, 1, 1),
            maturity_date=date(2029, 1, 1),
        )

        result = bond.price(curve, date(2024, 1, 1))

        # Price = 100 * exp(-0.05 * 5) ≈ 77.88
        expected = 100 * np.exp(-0.05 * 5)
        assert_almost_equal(result.clean_price, expected, decimal=1)


class TestSwaps:
    """Tests for interest rate swaps."""

    def test_vanilla_swap(self):
        """Test vanilla IRS pricing."""
        from finceptquantlib.instruments import InterestRateSwap, SwapType
        from finceptquantlib.curves import ZeroCurve

        curve = ZeroCurve(
            reference_date=date(2024, 1, 1),
            tenors=[0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
            values=[0.05] * 6,
        )

        swap = InterestRateSwap(
            notional=10_000_000,
            effective_date=date(2024, 1, 1),
            termination_date=date(2029, 1, 1),
            fixed_rate=0.05,
            swap_type=SwapType.PAYER,
        )

        # At par rate, NPV should be close to 0
        npv = swap.present_value(date(2024, 1, 1), curve)
        assert abs(npv) < 100_000  # Within 100k tolerance

        # Par rate should be close to 5%
        par = swap.par_rate(date(2024, 1, 1), curve)
        assert_almost_equal(par, 0.05, decimal=3)


class TestOptions:
    """Tests for interest rate options."""

    def test_cap_pricing(self):
        """Test cap pricing."""
        from finceptquantlib.instruments import Cap
        from finceptquantlib.curves import ZeroCurve

        curve = ZeroCurve(
            reference_date=date(2024, 1, 1),
            tenors=[0.25, 0.5, 1.0, 2.0, 3.0],
            values=[0.05] * 5,
        )

        cap = Cap(
            notional=10_000_000,
            strike=0.05,
            effective_date=date(2024, 1, 1),
            termination_date=date(2027, 1, 1),
        )

        price = cap.price(date(2024, 1, 1), curve, volatility=0.20)

        # Cap should have positive value
        assert price > 0


class TestCDS:
    """Tests for CDS pricing."""

    def test_cds_pricing(self):
        """Test CDS pricing."""
        from finceptquantlib.instruments import CreditDefaultSwap
        from finceptquantlib.instruments.fx_credit.credit import ConstantHazardRate
        from finceptquantlib.curves import ZeroCurve

        discount_curve = ZeroCurve(
            reference_date=date(2024, 1, 1),
            tenors=[1.0, 2.0, 3.0, 5.0],
            values=[0.05] * 4,
        )

        survival_curve = ConstantHazardRate(
            reference_date=date(2024, 1, 1),
            hazard_rate=0.02,  # 2% annual default probability
        )

        cds = CreditDefaultSwap(
            notional=10_000_000,
            effective_date=date(2024, 1, 1),
            maturity_date=date(2029, 1, 1),
            spread=0.01,  # 100bp running spread
        )

        # Par spread should be approximately hazard_rate * (1 - recovery)
        par = cds.par_spread(date(2024, 1, 1), discount_curve, survival_curve)
        expected_par = 0.02 * 0.6  # λ * (1-R)
        assert_almost_equal(par, expected_par, decimal=3)


class TestFX:
    """Tests for FX instruments."""

    def test_fx_forward(self):
        """Test FX forward pricing."""
        from finceptquantlib.instruments import FXForward
        from finceptquantlib.curves import ZeroCurve

        usd_curve = ZeroCurve(
            reference_date=date(2024, 1, 1),
            tenors=[0.25, 0.5, 1.0],
            values=[0.05, 0.05, 0.05],
        )

        eur_curve = ZeroCurve(
            reference_date=date(2024, 1, 1),
            tenors=[0.25, 0.5, 1.0],
            values=[0.03, 0.03, 0.03],
        )

        # Create forward from spot
        fwd = FXForward.from_spot(
            base_currency="EUR",
            quote_currency="USD",
            base_amount=1_000_000,
            spot_rate=1.10,
            value_date=date(2025, 1, 1),
            base_curve=eur_curve,
            quote_curve=usd_curve,
        )

        # Forward should be > spot (USD rates > EUR rates)
        assert fwd.forward_rate > 1.10

    def test_fx_option(self):
        """Test FX option pricing."""
        from finceptquantlib.instruments import FXOption
        from finceptquantlib.curves import ZeroCurve

        usd_curve = ZeroCurve(
            reference_date=date(2024, 1, 1),
            tenors=[0.5, 1.0],
            values=[0.05, 0.05],
        )

        eur_curve = ZeroCurve(
            reference_date=date(2024, 1, 1),
            tenors=[0.5, 1.0],
            values=[0.03, 0.03],
        )

        option = FXOption(
            base_currency="EUR",
            quote_currency="USD",
            notional=1_000_000,
            strike=1.10,
            expiry_date=date(2024, 7, 1),
            option_type="call",
        )

        price = option.price(
            valuation_date=date(2024, 1, 1),
            spot=1.10,
            domestic_curve=usd_curve,
            foreign_curve=eur_curve,
            volatility=0.10,
        )

        assert price > 0


# ============================================================================
# Risk Module Tests
# ============================================================================

class TestVaR:
    """Tests for VaR calculations."""

    def test_historical_var(self):
        """Test Historical VaR."""
        from finceptquantlib.risk import HistoricalVaR

        # Generate random returns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 250)

        var_calc = HistoricalVaR(
            returns=returns,
            portfolio_value=1_000_000,
            confidence_level=0.99,
        )

        result = var_calc.calculate()

        assert result.var > 0
        assert result.expected_shortfall >= result.var
        assert result.confidence_level == 0.99

    def test_parametric_var(self):
        """Test Parametric VaR."""
        from finceptquantlib.risk import ParametricVaR

        var_calc = ParametricVaR(
            portfolio_value=1_000_000,
            expected_return=0.0001,
            volatility=0.02,
            confidence_level=0.99,
        )

        result = var_calc.calculate()

        assert result.var > 0
        # At 99% confidence, VaR ≈ 2.33 * daily_vol * portfolio_value
        expected_var = 2.33 * 0.02 * 1_000_000
        assert abs(result.var - expected_var) < 5000


class TestStressTesting:
    """Tests for stress testing."""

    def test_scenario_analysis(self):
        """Test scenario analysis."""
        from finceptquantlib.risk import ScenarioAnalysis

        analyzer = ScenarioAnalysis(
            base_value=1_000_000,
            factor_deltas={
                "equity": -500_000,    # $500k loss per 100% equity drop
                "ir_curve": -100_000,  # $100k loss per 100bp rate rise
            },
        )

        shocks = {"equity": -0.20, "ir_curve": 0.01}
        impact = analyzer.calculate_impact(shocks)

        # Impact = -500k * (-0.20) + -100k * 0.01 = 100k - 1k = 99k
        expected = 500_000 * 0.20 - 100_000 * 0.01
        assert_almost_equal(impact, expected)

    def test_stress_test(self):
        """Test stress test framework."""
        from finceptquantlib.risk import StressTest, Scenario, ScenarioType

        def pricing_func(shocks):
            base = 1_000_000
            return base * (1 + shocks.get("equity", 0))

        stress = StressTest(
            base_value=1_000_000,
            pricing_func=pricing_func,
        )

        scenario = Scenario(
            name="Equity Crash",
            description="20% equity decline",
            scenario_type=ScenarioType.HYPOTHETICAL,
            shocks={"equity": -0.20},
        )

        result = stress.run_scenario(scenario)

        assert result.stressed_value == 800_000
        assert result.pnl_impact == -200_000


class TestPortfolio:
    """Tests for portfolio analytics."""

    def test_portfolio_risk(self):
        """Test portfolio risk metrics."""
        from finceptquantlib.risk import Portfolio, Position, AssetClassRisk

        portfolio = Portfolio(name="Test Portfolio")

        portfolio.add_position(Position(
            instrument_id="STOCK1",
            quantity=100,
            market_value=100_000,
            asset_class=AssetClassRisk.EQUITY,
        ))

        portfolio.add_position(Position(
            instrument_id="BOND1",
            quantity=50,
            market_value=50_000,
            asset_class=AssetClassRisk.FIXED_INCOME,
            duration=5.0,
        ))

        assert portfolio.total_market_value() == 150_000
        assert portfolio.num_positions == 2

        weights = portfolio.weights()
        assert_almost_equal(weights["STOCK1"], 100_000/150_000)

        # HHI should indicate some concentration
        hhi = portfolio.herfindahl_index()
        assert 0 < hhi < 1


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple modules."""

    def test_bond_portfolio_risk(self):
        """Test bond portfolio with risk analytics."""
        from finceptquantlib.instruments import FixedRateBond
        from finceptquantlib.curves import ZeroCurve
        from finceptquantlib.risk import Portfolio, Position, AssetClassRisk, HistoricalVaR
        import numpy as np

        # Create curve
        curve = ZeroCurve(
            reference_date=date(2024, 1, 1),
            tenors=[0.5, 1.0, 2.0, 5.0, 10.0],
            values=[0.05] * 5,
        )

        # Create bonds
        bond1 = FixedRateBond(
            notional=100,
            coupon_rate=0.05,
            issue_date=date(2024, 1, 1),
            maturity_date=date(2029, 1, 1),
        )

        bond2 = FixedRateBond(
            notional=100,
            coupon_rate=0.04,
            issue_date=date(2024, 1, 1),
            maturity_date=date(2034, 1, 1),
        )

        # Price bonds
        price1 = bond1.price(curve, date(2024, 1, 1))
        price2 = bond2.price(curve, date(2024, 1, 1))

        # Create portfolio
        portfolio = Portfolio()
        portfolio.add_position(Position(
            instrument_id="BOND1",
            quantity=1000,
            market_value=price1.clean_price * 1000,
            asset_class=AssetClassRisk.FIXED_INCOME,
            duration=price1.duration if hasattr(price1, 'duration') else 4.0,
        ))
        portfolio.add_position(Position(
            instrument_id="BOND2",
            quantity=500,
            market_value=price2.clean_price * 500,
            asset_class=AssetClassRisk.FIXED_INCOME,
            duration=7.0,
        ))

        # Portfolio should have positive value
        assert portfolio.total_market_value() > 0

        # Generate synthetic returns for VaR
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 250)

        var_calc = HistoricalVaR(
            returns=returns,
            portfolio_value=portfolio.total_market_value(),
            confidence_level=0.99,
        )

        var_result = var_calc.calculate()
        assert var_result.var > 0

    def test_swap_curve_sensitivity(self):
        """Test swap sensitivity to curve."""
        from finceptquantlib.instruments import InterestRateSwap, SwapType
        from finceptquantlib.curves import ZeroCurve
        from finceptquantlib.risk.measures.sensitivities import parallel_shift_sensitivity

        curve = ZeroCurve(
            reference_date=date(2024, 1, 1),
            tenors=[0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
            values=[0.05] * 8,
        )

        swap = InterestRateSwap(
            notional=10_000_000,
            effective_date=date(2024, 1, 1),
            termination_date=date(2029, 1, 1),
            fixed_rate=0.05,
            swap_type=SwapType.PAYER,
        )

        def price_with_curve(c):
            return swap.present_value(date(2024, 1, 1), c)

        # Calculate sensitivity to parallel shifts
        sensitivities = parallel_shift_sensitivity(price_with_curve, curve)

        # Payer swap should gain when rates rise
        assert sensitivities[10] > 0  # +10bp shift should increase value
        assert sensitivities[-10] < 0  # -10bp shift should decrease value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
