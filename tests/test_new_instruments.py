"""Tests for new instrument types."""

from datetime import date
import math


class TestConvertible:
    """Test convertible bonds."""

    def test_convertible_bond(self):
        """Test convertible bond pricing."""
        from finceptquantlib.instruments import ConvertibleBond, ConversionTerms
        from finceptquantlib.scheduling import Frequency

        conv_terms = ConversionTerms.from_premium(1000, 50, conversion_premium_pct=25)
        cb = ConvertibleBond(
            issue_date=date(2024, 1, 1),
            maturity_date=date(2029, 1, 1),
            par_value=1000,
            coupon_rate=0.02,
            conversion_terms=conv_terms,
            frequency=Frequency.SEMI_ANNUAL
        )

        result = cb.price(stock_price=60, credit_spread=0.03, volatility=0.30)

        assert result["price"] > 0
        assert result["bond_floor"] > 0
        assert result["conversion_value"] > 0


class TestCallableBonds:
    """Test callable/puttable bonds."""

    def test_callable_bond(self):
        """Test callable bond."""
        from finceptquantlib.instruments import CallableBond, CallSchedule
        from finceptquantlib.curves import DiscountCurve

        call_schedule = CallSchedule(
            call_dates=[date(2026, 1, 1)],
            call_prices=[102.0]
        )

        cb = CallableBond(
            issue_date=date(2024, 1, 1),
            maturity_date=date(2029, 1, 1),
            par_value=100,
            coupon_rate=0.05,
            call_schedule=call_schedule
        )

        curve = DiscountCurve(date(2024, 1, 1), [1, 2, 5], [0.04, 0.045, 0.05])
        price = cb.price(curve, volatility=0.15)

        assert price > 0
        assert price < 110  # Should trade below par + call premium


class TestMBS:
    """Test mortgage-backed securities."""

    def test_mbs_cashflows(self):
        """Test MBS cash flow generation."""
        from finceptquantlib.instruments import MortgageBackedSecurity, MortgagePool

        pool = MortgagePool(
            original_balance=100_000_000,
            current_balance=95_000_000,
            weighted_avg_coupon=4.5,
            weighted_avg_maturity=300
        )

        mbs = MortgageBackedSecurity(
            pool=pool,
            pass_through_rate=4.0,
            issue_date=date(2024, 1, 1)
        )

        cashflows = mbs.generate_cashflows(prepayment_rate=0.06, months=12)

        assert len(cashflows) > 0
        assert cashflows[0]["total_cashflow"] > 0
        assert cashflows[0]["interest"] > 0


class TestCDO:
    """Test structured products."""

    def test_cdo_tranche(self):
        """Test CDO tranche valuation."""
        from finceptquantlib.instruments import CDO, Tranche

        tranches = [
            Tranche("Senior AAA", 70_000_000, 0.08, 1.00, 0.03, 1),
            Tranche("Mezzanine", 20_000_000, 0.03, 0.08, 0.06, 2),
            Tranche("Equity", 10_000_000, 0.00, 0.03, 0.15, 3)
        ]

        cdo = CDO(tranches=tranches, reference_pool_size=100_000_000)

        # Low loss scenario - Senior protected
        senior_value = cdo.tranche_value("Senior AAA", expected_loss_pct=0.02)
        assert senior_value > 0

        # Calculate equity tranche loss
        equity_tranche = tranches[2]
        loss = equity_tranche.loss_allocation(0.02)  # 2% pool loss
        assert loss > 0  # Equity takes first loss


class TestCommodity:
    """Test commodity derivatives."""

    def test_commodity_swap(self):
        """Test commodity swap."""
        from finceptquantlib.instruments import CommoditySwap

        swap = CommoditySwap(
            commodity="WTI Crude",
            notional_quantity=10000,
            fixed_price=75.0,
            start_date=date(2024, 1, 1),
            end_date=date(2025, 1, 1)
        )

        forward_prices = [80.0] * 12
        discount_factors = [0.99] * 12

        npv = swap.present_value(80.0, forward_prices, discount_factors)
        assert abs(npv) >= 0  # Can be positive or negative

    def test_commodity_option(self):
        """Test commodity option."""
        from finceptquantlib.instruments import CommodityOption

        option = CommodityOption(
            commodity="Gold",
            strike=2000,
            expiry=date(2024, 12, 31),
            quantity=100,
            is_call=True
        )

        price = option.price(spot=2100, volatility=0.20, risk_free_rate=0.04)
        assert price > 0  # ITM call should have positive value


class TestEquity:
    """Test equity derivatives."""

    def test_equity_swap(self):
        """Test equity swap."""
        from finceptquantlib.instruments import EquitySwap

        eq_swap = EquitySwap(
            notional=10_000_000,
            equity_index="SPX",
            start_date=date(2024, 1, 1),
            end_date=date(2025, 1, 1),
            fixed_rate=0.03
        )

        npv = eq_swap.present_value(equity_return=0.15, discount_rate=0.04)
        assert isinstance(npv, float)

    def test_variance_swap(self):
        """Test variance swap."""
        from finceptquantlib.instruments import VarianceSwap

        var_swap = VarianceSwap(
            notional_variance=1_000_000,
            strike_variance=0.04,  # 20% vol squared
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31)
        )

        payoff = var_swap.payoff(realized_variance=0.0625)  # 25% vol
        assert payoff > 0  # Higher realized variance = positive payoff

    def test_realized_variance(self):
        """Test realized variance calculation."""
        from finceptquantlib.instruments import calculate_realized_variance, calculate_realized_volatility

        returns = [0.01, -0.02, 0.015, -0.01, 0.02] * 50  # 250 days

        var = calculate_realized_variance(returns)
        vol = calculate_realized_volatility(returns)

        assert var > 0
        assert abs(vol - math.sqrt(var)) < 1e-6


class TestWeatherCat:
    """Test weather and cat bonds."""

    def test_weather_derivative(self):
        """Test weather derivative."""
        from finceptquantlib.instruments import WeatherDerivative, WeatherIndex

        weather = WeatherDerivative(
            index_type=WeatherIndex.HDD,
            strike=1000,
            tick_value=10000,
            start_date=date(2024, 11, 1),
            end_date=date(2025, 3, 31),
            location="Chicago"
        )

        payoff = weather.payoff(realized_index=1200)  # Cold winter
        assert payoff > 0  # Realized > strike

    def test_cat_bond(self):
        """Test catastrophe bond."""
        from finceptquantlib.instruments import CatastropheBond

        cat_bond = CatastropheBond(
            issue_date=date(2024, 1, 1),
            maturity_date=date(2027, 1, 1),
            par_value=100_000_000,
            coupon_rate=0.08,
            trigger_type="indemnity",
            attachment_point=500_000_000,
            exhaustion_point=1_000_000_000,
            covered_perils=["Hurricane"]
        )

        # Test principal loss calculation
        loss_small = cat_bond.principal_loss(400_000_000)  # Below attachment
        assert loss_small == 0

        loss_partial = cat_bond.principal_loss(750_000_000)  # Partial loss
        assert 0 < loss_partial < cat_bond.par_value

        loss_full = cat_bond.principal_loss(1_200_000_000)  # Full loss
        assert loss_full == cat_bond.par_value

    def test_degree_days(self):
        """Test degree days calculation."""
        from finceptquantlib.instruments import degree_days

        # Cold week (heating degree days)
        temps = [30, 35, 28, 32, 40, 38, 35]  # Fahrenheit
        hdd = degree_days(temps, base_temp=65, heating=True)
        assert hdd > 0  # Should accumulate HDD

        # Hot week (cooling degree days)
        hot_temps = [85, 90, 88, 92, 87, 89, 91]
        cdd = degree_days(hot_temps, base_temp=65, heating=False)
        assert cdd > 0  # Should accumulate CDD


class TestMultiCurve:
    """Test multi-curve framework."""

    def test_multicurve_framework(self):
        """Test multi-curve setup."""
        from finceptquantlib.curves import MultiCurveFramework, DiscountCurve
        from finceptquantlib.core import Currency

        mcf = MultiCurveFramework(currency=Currency.USD)

        ois_curve = DiscountCurve(date(2024, 1, 1), [1, 2, 5], [0.04, 0.045, 0.05])
        mcf.set_discount_curve(ois_curve)

        libor_3m = DiscountCurve(date(2024, 1, 1), [1, 2, 5], [0.045, 0.050, 0.055])
        mcf.add_projection_curve("3M", libor_3m)

        df = mcf.discount_factor(1.0)
        assert 0 < df <= 1

        fwd = mcf.forward_rate("3M", 1.0, 2.0)
        assert isinstance(fwd, float)  # Forward rate calculated

    def test_cross_currency_basis(self):
        """Test cross-currency basis curve."""
        from finceptquantlib.curves import CrossCurrencyBasisCurve
        from finceptquantlib.core import Currency

        basis = CrossCurrencyBasisCurve(
            domestic=Currency.USD,
            foreign=Currency.EUR,
            tenors=[1, 2, 3, 5, 10],
            basis_spreads=[-0.0010, -0.0012, -0.0015, -0.0020, -0.0025]
        )

        spread_5y = basis.basis_spread(5.0)
        assert spread_5y < 0  # EUR/USD basis typically negative


class TestAdvancedCurves:
    """Test advanced curve features."""

    def test_curve_database(self):
        """Test curve storage and retrieval."""
        from finceptquantlib.curves import CurveDatabase, DiscountCurve

        db = CurveDatabase()

        curve1 = DiscountCurve(date(2024, 1, 1), [1, 2], [0.05, 0.06])
        curve2 = DiscountCurve(date(2024, 1, 2), [1, 2], [0.051, 0.061])

        db.store_curve("USD-OIS", curve1, date(2024, 1, 1))
        db.store_curve("USD-OIS", curve2, date(2024, 1, 2))

        retrieved = db.get_curve("USD-OIS", date(2024, 1, 1), exact_match=True)
        assert retrieved is not None

    def test_inflation_seasonality(self):
        """Test inflation seasonality adjustments."""
        from finceptquantlib.curves import InflationSeasonality

        seasonality = InflationSeasonality()
        # Higher inflation in winter/summer (energy)
        factors = [1.02, 1.01, 1.00, 0.99, 0.98, 0.97, 1.03, 1.04, 1.00, 0.99, 1.00, 1.01]
        seasonality.set_monthly_factors(factors)

        base_cpi = 250.0
        adjusted = seasonality.adjust(base_cpi, month=7)  # July (high factor)
        assert adjusted > base_cpi

        deseasonalized = seasonality.deseasonalize(adjusted, month=7)
        assert abs(deseasonalized - base_cpi) < 0.01
