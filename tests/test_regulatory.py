"""Tests for regulatory compliance module."""



class TestFRTB:
    """Test FRTB calculations."""

    def test_frtb_calculator(self):
        """Test FRTB-SA calculator."""
        from finceptquantlib.regulatory import FRTBSACalculator, RiskClass

        frtb = FRTBSACalculator()
        frtb.add_delta_sensitivity(RiskClass.INTEREST_RATE, bucket=1, sensitivity=1000000, risk_weight=0.017)
        frtb.add_delta_sensitivity(RiskClass.FX, bucket=1, sensitivity=500000, risk_weight=0.15)

        result = frtb.calculate()

        assert result.total_capital > 0
        assert result.delta_total >= 0
        assert RiskClass.INTEREST_RATE in result.risk_classes


class TestBasel:
    """Test Basel III/IV calculations."""

    def test_basel_calculator(self):
        """Test Basel capital ratios."""
        from finceptquantlib.regulatory import BaselCalculator, CreditExposure, AssetClass

        basel = BaselCalculator()
        basel.set_capital(cet1=500_000_000, tier1_additional=100_000_000, tier2=50_000_000)

        exposure = CreditExposure(
            exposure_amount=1_000_000_000,
            asset_class=AssetClass.CORPORATE,
            risk_weight=100.0
        )
        basel.add_credit_exposure(exposure)

        ratios = basel.calculate_ratios()

        assert ratios.cet1_ratio > 0
        assert ratios.tier1_ratio >= ratios.cet1_ratio
        assert ratios.total_capital_ratio >= ratios.tier1_ratio

    def test_credit_risk_weights(self):
        """Test standardized credit risk weights."""
        from finceptquantlib.regulatory import CreditExposure, AssetClass

        exp = CreditExposure(1_000_000, AssetClass.CORPORATE, risk_weight=100.0)
        rwa = exp.rwa_standardized()

        assert rwa == 1_000_000  # 100% risk weight


class TestIFRS9:
    """Test IFRS 9 / CECL."""

    def test_ecl_calculator(self):
        """Test ECL calculation."""
        from finceptquantlib.regulatory import (
            ECLCalculator,
            ProbabilityOfDefault,
            LossGivenDefault,
            ExposureAtDefault,
            Stage
        )

        pd = ProbabilityOfDefault([1.0, 2.0, 3.0], [0.01, 0.015, 0.02])
        lgd = LossGivenDefault(lgd_rate=0.45)
        ead = ExposureAtDefault(1_000_000, [1, 2, 3], [950_000, 900_000, 850_000], [0.05] * 3)

        ecl_calc = ECLCalculator(pd, lgd, ead)
        result = ecl_calc.calculate(Stage.STAGE_1)

        assert result.twelve_month_ecl > 0
        assert result.lifetime_ecl >= result.twelve_month_ecl
        assert result.stage == Stage.STAGE_1

    def test_staging(self):
        """Test stage assessment."""
        from finceptquantlib.regulatory import assess_stage, Stage

        # Stage 1: Performing
        stage = assess_stage(days_past_due=0, rating_downgrade=0, pd_increase_ratio=1.2)
        assert stage == Stage.STAGE_1

        # Stage 2: Significant increase
        stage = assess_stage(days_past_due=40, rating_downgrade=2, pd_increase_ratio=2.5)
        assert stage == Stage.STAGE_2

        # Stage 3: Credit-impaired
        stage = assess_stage(days_past_due=95, rating_downgrade=5, pd_increase_ratio=5.0)
        assert stage == Stage.STAGE_3


class TestSACCR:
    """Test SA-CCR."""

    def test_saccr_calculator(self):
        """Test SA-CCR EAD calculation."""
        from finceptquantlib.regulatory import SACCRCalculator, HedgingSet, AssetClassCCR

        saccr = SACCRCalculator()

        hs = HedgingSet(
            asset_class=AssetClassCCR.INTEREST_RATE,
            risk_factor="USD-SOFR",
            trades=["SWAP1", "SWAP2"],
            notionals=[10_000_000, -5_000_000],
            maturities=[5.0, 3.0],
            supervisory_deltas=[1.0, -1.0]
        )
        saccr.add_hedging_set(hs)

        ead = saccr.calculate_ead(mtm_value=50000, collateral=30000)

        assert ead > 0


class TestLiquidity:
    """Test liquidity ratios."""

    def test_lcr(self):
        """Test LCR calculation."""
        from finceptquantlib.regulatory import (
            LCRCalculator,
            HighQualityLiquidAssets,
            CashOutflows,
            CashInflows
        )

        hqla = HighQualityLiquidAssets(level_1=500_000_000, level_2a=200_000_000)
        outflows = CashOutflows(retail_deposits_stable=2_000_000_000, unsecured_wholesale=1_000_000_000)
        inflows = CashInflows(other_inflows=100_000_000)

        lcr_calc = LCRCalculator(hqla, outflows, inflows)
        lcr = lcr_calc.calculate()

        assert lcr > 0

    def test_nsfr(self):
        """Test NSFR calculation."""
        from finceptquantlib.regulatory import (
            NSFRCalculator,
            AvailableStableFunding,
            RequiredStableFunding
        )

        asf = AvailableStableFunding(capital=500_000_000, stable_deposits_retail=3_000_000_000)
        rsf = RequiredStableFunding(loans_retail=2_000_000_000, loans_corporate=1_500_000_000)

        nsfr_calc = NSFRCalculator(asf, rsf)
        nsfr = nsfr_calc.calculate()

        assert nsfr > 0


class TestStressTesting:
    """Test stress testing frameworks."""

    def test_ccar(self):
        """Test CCAR stress test."""
        from finceptquantlib.regulatory import CCARTest, SeverelyAdverseScenario

        scenario = SeverelyAdverseScenario.generate(num_quarters=9)
        ccar = CCARTest(initial_cet1=500_000_000, initial_rwa=5_000_000_000, scenario=scenario)

        result = ccar.run()

        # Stress scenario causes capital to decline (can go negative in severe stress)
        assert len(result.projections.quarters) == 9
        assert result.projections.cet1_ratios[0] < 10  # Capital deteriorates
        assert isinstance(result.passes, bool)

    def test_scenario_generation(self):
        """Test stress scenario generation."""
        from finceptquantlib.regulatory import SeverelyAdverseScenario, AdverseScenario, BaselineScenario

        severely_adverse = SeverelyAdverseScenario.generate()
        adverse = AdverseScenario.generate()
        baseline = BaselineScenario.generate()

        assert severely_adverse.num_quarters > 0
        assert adverse.cumulative_gdp_impact() > severely_adverse.cumulative_gdp_impact()
        assert baseline.cumulative_gdp_impact() > adverse.cumulative_gdp_impact()
