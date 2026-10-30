# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-10-30

First public release under the MIT license.

### Added
- `ExoticBarrier` alias for the equity barrier option and `CallableSwap` / `PuttableSwap`
  exports in `finceptquantlib.instruments`.
- Case-insensitive parsing for `OptionType` (`OptionType("CALL")`); `OptionType` and
  `BarrierType` now compare equal to their string values.
- Accuracy regression suite comparing distributions, option models, Fourier pricers,
  bond pricing, extreme value fits and calendars against SciPy and closed-form references.

### Fixed
- Heston semi-analytical pricing inverted the characteristic function at `ln(K/S)` and
  omitted the discount on the stock-measure term, mispricing every option; the FFT
  method used the wrong phase and crashed for grid sizes that were not a power of two.
- `lewis_fft` and `cos_method` returned incorrect prices for characteristic functions of
  `log(S_T)`; both now accept that convention consistently with `carr_madan_fft`.
- Regularized incomplete beta function returned wrong values, breaking Student-t and
  Beta CDFs and regression p-values; `FDistribution.pdf` was off by a factor of `x`.
- `t_cdf`, `t_ppf` and `StudentT.cdf` no longer switch to a normal approximation above
  100 degrees of freedom.
- Generalized Pareto MLE diverged on small-scale data; peaks-over-threshold VaR used an
  inverted exceedance ratio; GEV PWM fitting started from the wrong sign of the shape.
- Futures convexity adjustment used `t1 * (t2 - t1)` instead of `t1 * t2` (Ho-Lee) and an
  incorrect Hull-White expression.
- Fixed-rate and inflation-linked bond coupons accrued on business-day-adjusted dates;
  yield pricing used ACT/365F times instead of coupon periods, so par bonds did not
  price at par.
- Black-Scholes, Black-76 and Bachelier implied volatility failed to converge for deep
  in-the-money options; prices outside arbitrage bounds now raise `ValueError`.
- `BayesianChangepoint` could never report a changepoint; `hazard_rate` is now
  validated to lie in (0, 1).
- Dupire local volatility used a first-order time derivative.
- Logistic regression (IRLS) could diverge on collinear or separable data, giving results
  that depended on the Python version; it now regularises a singular Hessian and halves
  steps that would lower the likelihood.
- US federal calendar missed New Year's Day observed on December 31; UK calendar missed
  moved and one-off bank holidays.
- Monte Carlo routines reseeded Python's global random generator; they now use local
  generators.
- `from finceptquantlib import *` raised `AttributeError`; `__all__` lists are now
  complete, de-duplicated and free of private names.
- Docstring examples are executable and verified by the test suite.

### Changed
- Name clashes between subpackages resolve by a documented precedence at the top level;
  `finceptquantlib.CalibrationResult` and `finceptquantlib.BacktestResult` now refer to the
  solver and risk classes, and `finceptquantlib.black76_greeks` to `pricing`.
- Runtime dependencies reduced to `scipy` and `python-dateutil`.
