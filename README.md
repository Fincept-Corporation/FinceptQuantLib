# FinceptQuantLib

[![PyPI](https://img.shields.io/pypi/v/finceptquantlib.svg)](https://pypi.org/project/finceptquantlib/)
[![Python](https://img.shields.io/pypi/pyversions/finceptquantlib.svg)](https://pypi.org/project/finceptquantlib/)
[![CI](https://github.com/Fincept-Corporation/FinceptQuantLib/actions/workflows/ci.yml/badge.svg)](https://github.com/Fincept-Corporation/FinceptQuantLib/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Fincept-Corporation/FinceptQuantLib/blob/main/LICENSE)

Quantitative finance library from [Fincept Corporation](https://fincept.in): pricing, risk and
analytics for fixed income, derivatives and portfolios. Pure Python on top of SciPy, typed, and
validated against SciPy and closed-form references.

## Installation

```bash
pip install finceptquantlib
```

Requires Python 3.11+.

## Quick start

```python
from datetime import date
import finceptquantlib as fql

# Curves, swaps and bonds
curve = fql.ZeroCurve(date(2025, 1, 15), [0.5, 1, 2, 5, 10], [0.040, 0.041, 0.042, 0.043, 0.045])
swap = fql.InterestRateSwap(10_000_000, date(2025, 1, 15), date(2030, 1, 15), fixed_rate=0.042)
swap.par_rate(date(2025, 1, 15), curve)                                  # 0.04378

bond = fql.FixedRateBond(date(2025, 1, 15), date(2035, 1, 15), coupon_rate=0.045)
bond.price_from_yield(0.05, settlement_date=date(2025, 1, 15))           # 96.1027

# Options
call = fql.bs_call(S=100, K=105, r=0.03, q=0.01, sigma=0.25, T=1.0)      # 8.6124
fql.bs_implied_vol(call, S=100, K=105, r=0.03, q=0.01, T=1.0,
                   option_type=fql.OptionType.CALL)                      # 0.25

heston = fql.HestonModel(
    fql.HestonParameters(v0=0.04, kappa=1.5, theta=0.05, sigma=0.5, rho=-0.7),
    spot=100, rate=0.03, dividend=0.01,
)
heston.european_call_price(105, 1.0)                                     # 5.9313

# Portfolios and risk
cov = [[0.040, 0.006, 0.010], [0.006, 0.090, 0.012], [0.010, 0.012, 0.0625]]
weights = fql.max_sharpe_portfolio([0.08, 0.12, 0.10], cov, rf_rate=0.02).weights
fql.calculate_var(weights, cov, confidence=0.95, portfolio_value=1_000_000)  # 267,522
```

Every public name is available from the top-level package and from its subpackage
(`fql.ZeroCurve` is `finceptquantlib.curves.ZeroCurve`).

## Modules

| Module | Provides |
|---|---|
| `core` | Forward-mode automatic differentiation (`Dual`, `HyperDual`, gradients, Hessians), `Rate`, `Money`, `Tenor`, `Currency`, coupon periods and legs |
| `scheduling` | ISDA day counts, US / UK / TARGET / Japan calendars, business-day rules, schedule generation |
| `curves` | Discount, zero and forward curves, Nelson-Siegel(-Svensson), inflation and multi-curve frameworks, curve shifts |
| `instruments` | Fixed, floating, zero-coupon and inflation-linked bonds; IRS, OIS, basis, cross-currency and inflation swaps; FRAs, futures, caps/floors, swaptions; FX forwards and options; CDS; equity, commodity, convertible and exotic options |
| `pricing` | Black-Scholes, Black-76, Bachelier, Kirk and Margrabe formulas, Greeks, implied volatility, binomial trees |
| `models` | Heston, SABR, Merton, Kou and Variance Gamma, Dupire / SVI local volatility, Vasicek, CIR, Hull-White, Black-Karasinski |
| `volatility` | Volatility surfaces and cubes, SABR calibration, local volatility |
| `solver` | Curve bootstrapping, root finding, yield / spread / duration analytics |
| `risk` | Historical, parametric and Monte Carlo VaR / ES, backtesting, sensitivities, stress tests, XVA, copulas, extreme value theory |
| `portfolio` | Mean-variance optimisation, efficient frontier, Black-Litterman, risk parity, hierarchical risk parity |
| `regulatory` | Basel III/IV capital, SA-CCR, FRTB standardised approach, IFRS 9 ECL, LCR / NSFR, stress testing |
| `analysis` | Financial statement ratios, DuPont, earnings quality, DCF and residual income valuation, bank / insurer / REIT / utility analysis |
| `stochastic` | Brownian, mean-reverting, jump and Lévy processes; Euler-Maruyama, Milstein, exact and multilevel Monte Carlo simulation |
| `statistics` | Continuous and discrete distributions, ARIMA, GARCH, EGARCH, GJR-GARCH |
| `numerical` | Linear algebra, interpolation, quadrature, root finding, optimisation, FFT, Carr-Madan / Lewis / COS Fourier pricing |
| `ml` | Credit scorecards (WoE, logistic, calibration), model validation, regime and changepoint detection, ML curve and portfolio models |
| `economics` | Utility functions, normal-form games and auctions, general equilibrium |
| `physics` | Information theory (entropy, transfer entropy), statistical mechanics, thermodynamics |

## Development

```bash
git clone https://github.com/Fincept-Corporation/FinceptQuantLib.git
cd FinceptQuantLib
pip install -e ".[dev]"
pytest            # unit tests and docstring examples
ruff check src tests
```

See [CONTRIBUTING.md](https://github.com/Fincept-Corporation/FinceptQuantLib/blob/main/CONTRIBUTING.md) for guidelines and [CHANGELOG.md](https://github.com/Fincept-Corporation/FinceptQuantLib/blob/main/CHANGELOG.md) for release notes.

## License

MIT. See [LICENSE](https://github.com/Fincept-Corporation/FinceptQuantLib/blob/main/LICENSE).

FinceptQuantLib is provided for research and analytics. It is not investment advice; validate
results independently before relying on them for trading, risk or regulatory reporting.
