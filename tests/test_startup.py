"""Tests for startup valuation models."""



class TestVCMethod:
    """Test Venture Capital method."""

    def test_vc_basic(self):
        """Test basic VC valuation."""
        from finceptquantlib.analysis.valuation.startup import VentureCapitalMethod

        vc = VentureCapitalMethod(
            exit_value=100_000_000,
            years_to_exit=5,
            target_roi=10.0
        )

        result = vc.calculate(investment=2_000_000)

        assert result.pre_money_valuation > 0
        assert result.post_money_valuation == result.pre_money_valuation + 2_000_000
        assert 0 < result.equity_percentage < 100

    def test_vc_from_revenue(self):
        """Test VC method from revenue projections."""
        from finceptquantlib.analysis.valuation.startup import VentureCapitalMethod

        vc = VentureCapitalMethod.from_revenue_multiple(
            projected_revenue=50_000_000,
            revenue_multiple=3.0,
            years_to_exit=5,
            target_roi=10.0
        )

        result = vc.calculate(investment=5_000_000)
        assert result.pre_money_valuation > 0


class TestBerkusMethod:
    """Test Berkus method."""

    def test_berkus_basic(self):
        """Test Berkus valuation."""
        from finceptquantlib.analysis.valuation.startup import BerkusMethod

        berkus = BerkusMethod()
        berkus.set_score(
            sound_idea=400_000,
            prototype=300_000,
            quality_management=500_000,
            strategic_relationships=200_000,
            product_rollout=100_000
        )

        result = berkus.calculate()

        assert result.pre_money_valuation == 1_500_000
        assert result.method == "Berkus Method"

    def test_berkus_capping(self):
        """Test that scores are capped at max."""
        from finceptquantlib.analysis.valuation.startup import BerkusMethod

        berkus = BerkusMethod()
        berkus.set_score(
            sound_idea=1_000_000,  # Exceeds max
            prototype=600_000  # Exceeds max
        )

        result = berkus.calculate()
        # Should be capped at $500k each = $1M total
        assert result.pre_money_valuation == 1_000_000


class TestScorecardMethod:
    """Test Scorecard valuation."""

    def test_scorecard_basic(self):
        """Test scorecard method."""
        from finceptquantlib.analysis.valuation.startup import ScorecardValuation

        scorecard = ScorecardValuation(
            comparable_avg_valuation=3_000_000,
            region="US-California",
            industry="SaaS"
        )

        scorecard.set_factors(
            management_team=1.2,  # 20% better than comparables
            size_of_opportunity=1.3,  # 30% better
            product_technology=1.1
        )

        result = scorecard.calculate()

        # Should be higher than base due to better factors
        assert result.pre_money_valuation > 3_000_000


class TestRiskFactorSummation:
    """Test Risk Factor Summation."""

    def test_risk_factor_basic(self):
        """Test risk factor summation."""
        from finceptquantlib.analysis.valuation.startup import RiskFactorSummation, RiskLevel

        rfs = RiskFactorSummation(base_valuation=2_000_000)
        rfs.assess_risks(
            management=RiskLevel.LOW,  # +$250k
            technology=RiskLevel.MEDIUM,  # $0
            competition=RiskLevel.HIGH  # -$250k
        )

        result = rfs.calculate()

        # Base $2M + $250k - $250k = $2M
        assert result.pre_money_valuation == 2_000_000


class TestFirstChicago:
    """Test First Chicago method."""

    def test_first_chicago(self):
        """Test scenario-based valuation."""
        from finceptquantlib.analysis.valuation.startup import FirstChicagoMethod

        fc = FirstChicagoMethod(discount_rate=0.30)

        fc.add_scenario("Best", 0.30, revenue=50_000_000, ebitda_margin=0.25,
                       exit_multiple=12, years_to_exit=5)
        fc.add_scenario("Base", 0.50, revenue=20_000_000, ebitda_margin=0.20,
                       exit_multiple=8, years_to_exit=5)
        fc.add_scenario("Worst", 0.20, revenue=5_000_000, ebitda_margin=0.10,
                       exit_multiple=5, years_to_exit=5)

        result = fc.calculate()

        assert result.pre_money_valuation > 0
        assert len(result.assumptions["scenarios"]) == 3


class TestOptionPricing:
    """Test option pricing method."""

    def test_option_pricing(self):
        """Test option pricing valuation."""
        from finceptquantlib.analysis.valuation.startup import OptionPricingValuation

        option_val = OptionPricingValuation(
            total_addressable_market=1_000_000_000,
            cost_to_capture=50_000_000,
            time_to_profitability=3.0,
            volatility=0.80
        )

        result = option_val.calculate(market_share_capture=0.10)

        assert result.pre_money_valuation > 0
        # Should be less than TAM * market share
        assert result.pre_money_valuation < 100_000_000


class TestMilestoneValuation:
    """Test milestone-based valuation."""

    def test_milestone_basic(self):
        """Test milestone valuation."""
        from finceptquantlib.analysis.valuation.startup import MilestoneValuation

        milestone_val = MilestoneValuation(base_value=500_000)

        milestone_val.add_milestone(
            "MVP Launch",
            value_created=1_000_000,
            completion=75,  # 75% complete
            probability=0.90
        )

        milestone_val.add_milestone(
            "First Revenue",
            value_created=2_000_000,
            completion=30,  # 30% complete
            probability=0.70
        )

        result = milestone_val.calculate(risk_adjustment=0.8)

        # Base + weighted milestones
        assert result.pre_money_valuation > 500_000


class TestNetworkEffects:
    """Test network effects valuation."""

    def test_network_valuation(self):
        """Test network effects valuation."""
        from finceptquantlib.analysis.valuation.startup import NetworkEffectsValuation

        network_val = NetworkEffectsValuation(
            current_users=10_000,
            value_per_user=50,
            network_coefficient=2.0
        )

        result = network_val.calculate(projected_users_1yr=100_000)

        # Should increase significantly with network effects
        assert result.pre_money_valuation > 0


class TestUtilityFunctions:
    """Test utility functions."""

    def test_valuation_range(self):
        """Test typical valuation ranges."""
        from finceptquantlib.analysis.valuation.startup import pre_revenue_valuation_range, StartupStage

        low, high = pre_revenue_valuation_range(StartupStage.PROTOTYPE, "Technology", "US")

        assert low > 0
        assert high > low
        assert high <= 10_000_000  # Reasonable for prototype stage

    def test_dilution_calc(self):
        """Test ownership dilution calculation."""
        from finceptquantlib.analysis.valuation.startup import calculate_ownership_dilution

        dilution = calculate_ownership_dilution(
            pre_money=4_000_000,
            investment=1_000_000,
            option_pool_pct=0.15
        )

        assert "founders" in dilution
        assert "investors" in dilution
        assert "option_pool" in dilution

        # Total should be ~100%
        total = dilution["founders"] + dilution["investors"] + dilution["option_pool"]
        assert abs(total - 100) < 0.01
