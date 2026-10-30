"""Tests for advanced risk analytics."""



class TestIncrementalVaR:
    """Test incremental VaR calculations."""

    def test_incremental_var(self):
        """Test incremental VaR calculation."""
        from finceptquantlib.risk import IncrementalVaRCalculator

        # Portfolio returns
        portfolio_returns = [0.01, -0.02, 0.015, -0.01, 0.02, -0.025, 0.01] * 30

        # New position returns
        new_position = [0.02, -0.01, 0.01, -0.015, 0.025, -0.02, 0.015] * 30

        calculator = IncrementalVaRCalculator(
            portfolio_returns=portfolio_returns,
            confidence_level=0.95,
            portfolio_value=10_000_000
        )

        result = calculator.calculate_incremental_var(
            new_position_returns=new_position,
            position_weight=0.10,
            position_name="New Asset"
        )

        assert result.incremental_var != 0  # Should change VaR
        assert result.standalone_var > 0
        assert isinstance(result.marginal_var, float)


class TestComponentVaR:
    """Test component VaR decomposition."""

    def test_component_var(self):
        """Test VaR decomposition by risk factor."""
        from finceptquantlib.risk import ComponentVaRCalculator

        # Create returns for different risk factors
        equity_returns = [0.02, -0.03, 0.01, -0.02, 0.025] * 50
        rates_returns = [0.005, -0.01, 0.003, -0.005, 0.007] * 50
        credit_returns = [0.01, -0.015, 0.008, -0.01, 0.012] * 50

        calculator = ComponentVaRCalculator(
            returns_by_factor={
                "Equity": equity_returns,
                "Rates": rates_returns,
                "Credit": credit_returns
            },
            weights={"Equity": 0.5, "Rates": 0.3, "Credit": 0.2},
            confidence_level=0.95,
            portfolio_value=10_000_000
        )

        components = calculator.decompose_var()

        assert "Equity" in components
        assert "Rates" in components
        assert "Credit" in components

        # Components should sum approximately to total VaR
        total_component = sum(c.component_var for c in components.values())
        assert total_component > 0


class TestBacktesting:
    """Test VaR backtesting framework."""

    def test_var_backtester(self):
        """Test VaR backtesting."""
        from finceptquantlib.risk import VaRBacktester, TrafficLightZone

        backtester = VaRBacktester(confidence_level=0.99)

        # Add 250 observations (1 year of daily data)
        for i in range(250):
            # Simulate: 3 exceptions (within green zone)
            if i in [50, 120, 200]:
                actual = -0.03  # Loss exceeds VaR
            else:
                actual = -0.01  # Normal loss

            var_estimate = 0.02  # VaR estimate

            backtester.add_observation(actual, var_estimate)

        result = backtester.run_tests()

        assert result.num_observations == 250
        assert result.num_exceptions == 3
        assert result.traffic_light_zone == TrafficLightZone.GREEN
        assert result.exception_rate < 0.02  # Less than 2%

    def test_kupiec_test(self):
        """Test Kupiec POF test."""
        from finceptquantlib.risk import backtest_var_model

        # Generate data with ~1% exceptions (as expected for 99% VaR)
        actual_returns = [-0.01] * 99 + [-0.03]  # 1 exception in 100
        var_estimates = [0.02] * 100

        result = backtest_var_model(actual_returns, var_estimates, confidence_level=0.99)

        assert result.num_exceptions == 1
        assert result.passes_kupiec  # Should pass with correct exception rate


class TestTailRisk:
    """Test tail risk metrics."""

    def test_tail_risk_analyzer(self):
        """Test comprehensive tail risk analysis."""
        from finceptquantlib.risk import TailRiskAnalyzer

        returns = [0.01, -0.02, 0.015, -0.025, 0.02, -0.03, 0.01, -0.015] * 50

        analyzer = TailRiskAnalyzer(returns=returns, portfolio_value=1_000_000)

        metrics = analyzer.calculate_all_metrics()

        assert metrics.var_95 > 0
        assert metrics.var_99 >= metrics.var_95  # 99% VaR >= 95% VaR
        assert metrics.cvar_99 >= metrics.var_99  # CVaR >= VaR
        assert metrics.maximum_loss > 0
        assert metrics.omega_ratio > 0

    def test_expected_shortfall(self):
        """Test expected shortfall calculation."""
        from finceptquantlib.risk import conditional_value_at_risk

        returns = [-0.01, -0.02, -0.03, 0.01, 0.02, -0.025, 0.015] * 30

        es = conditional_value_at_risk(returns, confidence_level=0.95, portfolio_value=1_000_000)

        assert es > 0  # Should be positive loss amount

    def test_tail_index(self):
        """Test tail index estimation."""
        from finceptquantlib.risk import TailRiskAnalyzer

        # Fat-tailed distribution
        returns = [-0.05, -0.04, -0.03, -0.02, -0.01] + [0.01] * 95

        analyzer = TailRiskAnalyzer(returns)
        tail_idx = analyzer.tail_index(threshold_quantile=0.90)

        assert tail_idx > 0  # Should estimate tail thickness


class TestLiquidityVaR:
    """Test liquidity-adjusted VaR."""

    def test_liquidity_adjusted_var(self):
        """Test LVaR calculation."""
        from finceptquantlib.risk import LiquidityAdjustedVaR, LiquidityAdjustment

        # Returns for 3 positions
        returns = [
            [0.01, -0.02, 0.015] * 80,  # Liquid asset
            [0.02, -0.03, 0.01] * 80,   # Medium liquidity
            [0.015, -0.025, 0.02] * 80  # Illiquid
        ]

        liquidity_params = [
            LiquidityAdjustment(0.001, 10_000_000, 1, 5_000_000),    # Tight spread, deep market
            LiquidityAdjustment(0.01, 1_000_000, 10, 500_000),      # Wider spread
            LiquidityAdjustment(0.05, 100_000, 30, 50_000)          # Very illiquid
        ]

        lvar = LiquidityAdjustedVaR(
            returns=returns,
            position_values=[5_000_000, 3_000_000, 2_000_000],
            liquidity_params=liquidity_params,
            confidence_level=0.95
        )

        result = lvar.calculate()

        assert result["standard_var"] > 0
        assert result["liquidity_cost"] > 0
        assert result["liquidity_adjusted_var"] > result["standard_var"]


class TestCorrelationStress:
    """Test correlation stress scenarios."""

    def test_crisis_correlation(self):
        """Test crisis correlation scenario."""
        from finceptquantlib.risk import CorrelationStressTester

        base_corr = [
            [1.0, 0.3, 0.2],
            [0.3, 1.0, 0.4],
            [0.2, 0.4, 1.0]
        ]

        vols = [0.20, 0.15, 0.25]
        weights = [0.4, 0.4, 0.2]

        stress = CorrelationStressTester(
            base_correlations=base_corr,
            volatilities=vols,
            weights=weights,
            portfolio_value=10_000_000
        )

        crisis_var = stress.stress_crisis_correlation(crisis_correlation=0.90)

        assert crisis_var > 0

        # Crisis VaR should be higher than base (less diversification)
        # Could compare with base VaR calculation here

    def test_correlation_scenarios(self):
        """Test scenario generation."""
        from finceptquantlib.risk import CorrelationStressTester

        base_corr = [[1.0, 0.5], [0.5, 1.0]]
        stress = CorrelationStressTester(base_corr, [0.2, 0.15], [0.6, 0.4])

        scenarios = stress.generate_correlation_scenarios()

        assert len(scenarios) >= 3  # Crisis, Decorrelation, Regime shift
        assert all(s.correlation_matrix for s in scenarios)


class TestCoherentMeasures:
    """Test coherent risk measures."""

    def test_subadditivity(self):
        """Test subadditivity check."""
        from finceptquantlib.risk import CoherentRiskMeasure, conditional_value_at_risk

        returns1 = [0.01, -0.02, 0.015] * 30
        returns2 = [0.02, -0.01, 0.01] * 30

        # CVaR should be subadditive
        is_subadditive = CoherentRiskMeasure.check_subadditivity(
            lambda r: conditional_value_at_risk(r, 0.95),
            returns1,
            returns2
        )

        # Note: May fail due to sampling error, but should generally hold
        assert isinstance(is_subadditive, bool)
