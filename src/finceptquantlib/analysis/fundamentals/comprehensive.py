"""
Comprehensive Financial Analysis

Unified interface for complete financial statement analysis:
- Ratio Summary
- Analysis Report Generation
- Multi-Period Analysis
- Executive Summary
- Export Capabilities

The FinancialAnalyzer class provides a single interface for all analysis modules.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from .statements import (
    FinancialStatements
)
from .valuation import MarketData

# Import all analysis modules
from .profitability import ProfitabilityAnalysis, ProfitabilityResult
from .liquidity import LiquidityAnalysis, LiquidityResult
from .solvency import SolvencyAnalysis, SolvencyResult
from .efficiency import EfficiencyAnalysis, EfficiencyResult
from .valuation import ValuationAnalysis, ValuationResult
from .cashflow import CashFlowAnalysis, CashFlowResult
from .dupont import DuPontAnalysis, DuPont5Result
from .quality import EarningsQualityAnalysis, EarningsQualityResult
from .growth import GrowthAnalysis, GrowthResult
from ..valuation.predictive import (
    altman_z_prime, piotroski_f_score, beneish_m_score,
    AltmanZResult, PiotroskiFResult, BeneishMResult
)


@dataclass
class RatioSummary:
    """Summary of all key financial ratios."""
    # Profitability
    gross_margin: float = 0.0
    operating_margin: float = 0.0
    net_margin: float = 0.0
    roe: float = 0.0
    roa: float = 0.0
    roic: float = 0.0

    # Liquidity
    current_ratio: float = 0.0
    quick_ratio: float = 0.0
    cash_ratio: float = 0.0
    cash_conversion_cycle: float = 0.0

    # Solvency
    debt_to_equity: float = 0.0
    debt_to_assets: float = 0.0
    interest_coverage: float = 0.0
    net_debt_to_ebitda: float = 0.0

    # Efficiency
    asset_turnover: float = 0.0
    inventory_turnover: float = 0.0
    receivables_turnover: float = 0.0

    # Growth
    revenue_growth: float = 0.0
    earnings_growth: float = 0.0
    sustainable_growth_rate: float = 0.0

    # Valuation (if market data available)
    pe_ratio: float = 0.0
    pb_ratio: float = 0.0
    ev_ebitda: float = 0.0

    # Quality
    earnings_quality_score: float = 0.0
    cash_conversion_quality: float = 0.0


@dataclass
class AnalysisReport:
    """Complete financial analysis report."""
    company_name: str = ""
    analysis_date: str = ""

    # Summary metrics
    summary: RatioSummary = None

    # Detailed results
    profitability: ProfitabilityResult = None
    liquidity: LiquidityResult = None
    solvency: SolvencyResult = None
    efficiency: EfficiencyResult = None
    valuation: ValuationResult = None
    cash_flow: CashFlowResult = None
    dupont: DuPont5Result = None
    quality: EarningsQualityResult = None
    growth: GrowthResult = None

    # Predictive models
    altman_z: AltmanZResult = None
    piotroski_f: PiotroskiFResult = None
    beneish_m: BeneishMResult = None

    # Overall assessment
    overall_rating: str = ""
    strengths: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)


class FinancialAnalyzer:
    """
    Unified financial statement analyzer.

    Provides a single interface for comprehensive financial analysis.

    Example:
        # Create analyzer
        analyzer = FinancialAnalyzer(
            current_statements=statements,
            prior_statements=prior_statements,
            market_data=market_data,
            company_name="Apple Inc."
        )

        # Get complete report
        report = analyzer.full_analysis()

        # Get specific analysis
        profitability = analyzer.profitability()
        dupont = analyzer.dupont()

        # Get text summary
        print(analyzer.executive_summary())

        # Export to dictionary
        data = analyzer.to_dict()
    """

    def __init__(
        self,
        current_statements: FinancialStatements,
        prior_statements: Optional[FinancialStatements] = None,
        market_data: Optional[MarketData] = None,
        company_name: str = "Company"
    ):
        self.current = current_statements
        self.prior = prior_statements
        self.market = market_data
        self.company_name = company_name

        # Cache results
        self._profitability = None
        self._liquidity = None
        self._solvency = None
        self._efficiency = None
        self._valuation = None
        self._cash_flow = None
        self._dupont = None
        self._quality = None
        self._growth = None

    # =========================================================================
    # INDIVIDUAL ANALYSIS METHODS
    # =========================================================================

    def profitability(self) -> ProfitabilityResult:
        """Run profitability analysis."""
        if self._profitability is None:
            analysis = ProfitabilityAnalysis(self.current, self.prior)
            self._profitability = analysis.analyze()
        return self._profitability

    def liquidity(self) -> LiquidityResult:
        """Run liquidity analysis."""
        if self._liquidity is None:
            analysis = LiquidityAnalysis(self.current, self.prior)
            self._liquidity = analysis.analyze()
        return self._liquidity

    def solvency(self) -> SolvencyResult:
        """Run solvency analysis."""
        if self._solvency is None:
            analysis = SolvencyAnalysis(self.current, self.prior)
            self._solvency = analysis.analyze()
        return self._solvency

    def efficiency(self) -> EfficiencyResult:
        """Run efficiency analysis."""
        if self._efficiency is None:
            analysis = EfficiencyAnalysis(self.current, self.prior)
            self._efficiency = analysis.analyze()
        return self._efficiency

    def valuation(self) -> Optional[ValuationResult]:
        """Run valuation analysis (requires market data)."""
        if self.market is None:
            return None
        if self._valuation is None:
            analysis = ValuationAnalysis(self.current, self.market, self.prior)
            self._valuation = analysis.analyze()
        return self._valuation

    def cash_flow(self) -> CashFlowResult:
        """Run cash flow analysis."""
        if self._cash_flow is None:
            analysis = CashFlowAnalysis(self.current, self.prior)
            self._cash_flow = analysis.analyze()
        return self._cash_flow

    def dupont(self) -> DuPont5Result:
        """Run DuPont analysis."""
        if self._dupont is None:
            analysis = DuPontAnalysis(self.current, self.prior)
            self._dupont = analysis.five_factor()
        return self._dupont

    def quality(self) -> EarningsQualityResult:
        """Run earnings quality analysis."""
        if self._quality is None:
            analysis = EarningsQualityAnalysis(self.current, self.prior)
            self._quality = analysis.analyze()
        return self._quality

    def growth(self) -> GrowthResult:
        """Run growth analysis."""
        if self._growth is None:
            analysis = GrowthAnalysis(self.current, self.prior)
            self._growth = analysis.analyze()
        return self._growth

    # =========================================================================
    # PREDICTIVE MODELS
    # =========================================================================

    def altman_z(self) -> AltmanZResult:
        """Calculate Altman Z-Score (private company version)."""
        return altman_z_prime(
            self.current.income_statement,
            self.current.balance_sheet
        )

    def piotroski_f(self) -> Optional[PiotroskiFResult]:
        """Calculate Piotroski F-Score (requires prior period)."""
        if self.prior is None:
            return None
        return piotroski_f_score(self.current, self.prior)

    def beneish_m(self) -> Optional[BeneishMResult]:
        """Calculate Beneish M-Score (requires prior period)."""
        if self.prior is None:
            return None
        return beneish_m_score(self.current, self.prior)

    # =========================================================================
    # SUMMARY AND REPORTS
    # =========================================================================

    def ratio_summary(self) -> RatioSummary:
        """Get summary of key ratios."""
        prof = self.profitability()
        liq = self.liquidity()
        solv = self.solvency()
        eff = self.efficiency()
        qual = self.quality()
        grw = self.growth()
        val = self.valuation()

        return RatioSummary(
            # Profitability
            gross_margin=prof.gross_margin,
            operating_margin=prof.operating_margin,
            net_margin=prof.net_margin,
            roe=prof.return_on_equity,
            roa=prof.return_on_assets,
            roic=prof.return_on_invested_capital,

            # Liquidity
            current_ratio=liq.current_ratio,
            quick_ratio=liq.quick_ratio,
            cash_ratio=liq.cash_ratio,
            cash_conversion_cycle=liq.cash_conversion_cycle,

            # Solvency
            debt_to_equity=solv.debt_to_equity,
            debt_to_assets=solv.debt_to_assets,
            interest_coverage=solv.interest_coverage,
            net_debt_to_ebitda=solv.net_debt_to_ebitda,

            # Efficiency
            asset_turnover=eff.asset_turnover,
            inventory_turnover=eff.inventory_turnover,
            receivables_turnover=eff.receivables_turnover,

            # Growth
            revenue_growth=grw.revenue_growth,
            earnings_growth=grw.earnings_growth,
            sustainable_growth_rate=grw.sustainable_growth_rate,

            # Valuation
            pe_ratio=val.price_to_earnings if val else 0.0,
            pb_ratio=val.price_to_book if val else 0.0,
            ev_ebitda=val.ev_to_ebitda if val else 0.0,

            # Quality
            earnings_quality_score=qual.overall_quality_score,
            cash_conversion_quality=qual.cash_conversion_quality if isinstance(qual.cash_conversion_quality, float) and qual.cash_conversion_quality != float('inf') else 0.0,
        )

    def full_analysis(self) -> AnalysisReport:
        """Generate complete analysis report."""
        summary = self.ratio_summary()

        # Collect strengths and concerns
        strengths = []
        concerns = []

        # Profitability assessment
        if summary.roe > 0.15:
            strengths.append("Strong ROE (>15%)")
        elif summary.roe < 0.08:
            concerns.append("Weak ROE (<8%)")

        if summary.net_margin > 0.10:
            strengths.append("Healthy profit margins")
        elif summary.net_margin < 0.03:
            concerns.append("Low profit margins")

        # Liquidity assessment
        if summary.current_ratio >= 2.0:
            strengths.append("Strong liquidity position")
        elif summary.current_ratio < 1.0:
            concerns.append("Potential liquidity concerns")

        # Solvency assessment
        if summary.debt_to_equity < 0.5:
            strengths.append("Conservative debt levels")
        elif summary.debt_to_equity > 2.0:
            concerns.append("High leverage")

        if summary.interest_coverage >= 5.0:
            strengths.append("Strong interest coverage")
        elif summary.interest_coverage < 2.0:
            concerns.append("Tight interest coverage")

        # Efficiency assessment
        if summary.cash_conversion_cycle < 60:
            strengths.append("Efficient working capital management")
        elif summary.cash_conversion_cycle > 120:
            concerns.append("Long cash conversion cycle")

        # Quality assessment
        if summary.earnings_quality_score >= 70:
            strengths.append("High earnings quality")
        elif summary.earnings_quality_score < 50:
            concerns.append("Earnings quality concerns")

        # Overall rating
        score = len(strengths) - len(concerns)
        if score >= 3:
            overall = "STRONG"
        elif score >= 1:
            overall = "GOOD"
        elif score >= -1:
            overall = "AVERAGE"
        else:
            overall = "WEAK"

        return AnalysisReport(
            company_name=self.company_name,
            summary=summary,
            profitability=self.profitability(),
            liquidity=self.liquidity(),
            solvency=self.solvency(),
            efficiency=self.efficiency(),
            valuation=self.valuation(),
            cash_flow=self.cash_flow(),
            dupont=self.dupont(),
            quality=self.quality(),
            growth=self.growth(),
            altman_z=self.altman_z(),
            piotroski_f=self.piotroski_f(),
            beneish_m=self.beneish_m(),
            overall_rating=overall,
            strengths=strengths,
            concerns=concerns,
        )

    def executive_summary(self) -> str:
        """Generate executive summary text."""
        report = self.full_analysis()
        summary = report.summary

        text = f"FINANCIAL ANALYSIS: {self.company_name}\n"
        text += "=" * 60 + "\n\n"

        text += f"OVERALL RATING: {report.overall_rating}\n\n"

        # Key Metrics
        text += "KEY METRICS:\n"
        text += "-" * 40 + "\n"
        text += f"  Revenue Growth:     {summary.revenue_growth:>8.1%}\n"
        text += f"  Net Margin:         {summary.net_margin:>8.1%}\n"
        text += f"  ROE:                {summary.roe:>8.1%}\n"
        text += f"  ROIC:               {summary.roic:>8.1%}\n"
        text += f"  Debt/Equity:        {summary.debt_to_equity:>8.2f}x\n"
        text += f"  Current Ratio:      {summary.current_ratio:>8.2f}x\n"

        if summary.pe_ratio > 0:
            text += f"  P/E Ratio:          {summary.pe_ratio:>8.1f}x\n"
        if summary.ev_ebitda > 0:
            text += f"  EV/EBITDA:          {summary.ev_ebitda:>8.1f}x\n"

        text += "\n"

        # DuPont Decomposition
        dp = report.dupont
        text += "ROE DECOMPOSITION (DuPont):\n"
        text += "-" * 40 + "\n"
        text += f"  Tax Burden:         {dp.tax_burden:>8.2%}\n"
        text += f"  Interest Burden:    {dp.interest_burden:>8.2%}\n"
        text += f"  EBIT Margin:        {dp.ebit_margin:>8.2%}\n"
        text += f"  Asset Turnover:     {dp.asset_turnover:>8.2f}x\n"
        text += f"  Financial Leverage: {dp.equity_multiplier:>8.2f}x\n"
        text += f"  = ROE:              {dp.roe:>8.2%}\n\n"

        # Predictive Models
        text += "FINANCIAL HEALTH INDICATORS:\n"
        text += "-" * 40 + "\n"
        if report.altman_z:
            text += f"  Altman Z-Score:     {report.altman_z.z_score:>8.2f} ({report.altman_z.zone})\n"
        if report.piotroski_f:
            text += f"  Piotroski F-Score:  {report.piotroski_f.f_score:>8d}/9 ({report.piotroski_f.category})\n"
        if report.beneish_m:
            manip = "Likely" if report.beneish_m.manipulation_likely else "Unlikely"
            text += f"  Beneish M-Score:    {report.beneish_m.m_score:>8.2f} (Manipulation: {manip})\n"

        text += f"  Earnings Quality:   {summary.earnings_quality_score:>8.0f}/100\n\n"

        # Strengths and Concerns
        if report.strengths:
            text += "STRENGTHS:\n"
            for s in report.strengths:
                text += f"  + {s}\n"
            text += "\n"

        if report.concerns:
            text += "CONCERNS:\n"
            for c in report.concerns:
                text += f"  - {c}\n"
            text += "\n"

        # Conclusion
        text += "CONCLUSION:\n"
        text += "-" * 40 + "\n"

        if report.overall_rating == "STRONG":
            text += "The company demonstrates strong financial health across\n"
            text += "profitability, liquidity, and solvency metrics.\n"
        elif report.overall_rating == "GOOD":
            text += "The company shows solid financial performance with\n"
            text += "some areas that could be improved.\n"
        elif report.overall_rating == "AVERAGE":
            text += "The company's financial metrics are mixed, with\n"
            text += "both strengths and areas requiring attention.\n"
        else:
            text += "The company's financial position shows several\n"
            text += "areas of concern that warrant close monitoring.\n"

        return text

    def to_dict(self) -> Dict[str, Any]:
        """Export all analysis results as dictionary."""
        summary = self.ratio_summary()

        return {
            'company_name': self.company_name,
            'summary': {
                'gross_margin': summary.gross_margin,
                'operating_margin': summary.operating_margin,
                'net_margin': summary.net_margin,
                'roe': summary.roe,
                'roa': summary.roa,
                'roic': summary.roic,
                'current_ratio': summary.current_ratio,
                'quick_ratio': summary.quick_ratio,
                'cash_ratio': summary.cash_ratio,
                'debt_to_equity': summary.debt_to_equity,
                'interest_coverage': summary.interest_coverage,
                'asset_turnover': summary.asset_turnover,
                'revenue_growth': summary.revenue_growth,
                'earnings_growth': summary.earnings_growth,
                'pe_ratio': summary.pe_ratio,
                'ev_ebitda': summary.ev_ebitda,
                'earnings_quality_score': summary.earnings_quality_score,
            },
            'profitability': ProfitabilityAnalysis(self.current, self.prior).to_dict(),
            'liquidity': LiquidityAnalysis(self.current, self.prior).to_dict(),
            'solvency': SolvencyAnalysis(self.current, self.prior).to_dict(),
            'efficiency': EfficiencyAnalysis(self.current, self.prior).to_dict(),
            'cash_flow': CashFlowAnalysis(self.current, self.prior).to_dict(),
            'dupont': DuPontAnalysis(self.current, self.prior).to_dict(),
            'quality': EarningsQualityAnalysis(self.current, self.prior).to_dict(),
            'growth': GrowthAnalysis(self.current, self.prior).to_dict(),
            'altman_z': {
                'score': self.altman_z().z_score,
                'zone': self.altman_z().zone,
            },
        }
