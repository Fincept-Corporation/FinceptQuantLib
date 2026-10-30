"""Accuracy regression tests against SciPy and closed-form references."""

import cmath
import math
import random
from datetime import date

import pytest
from scipy import integrate, special, stats

import finceptquantlib as fql
from finceptquantlib.core.foundations.types import OptionType
from finceptquantlib.core.utilities import distributions as dist
from finceptquantlib.models.stochastic_vol import HestonModel, HestonParameters
from finceptquantlib.numerical import cos_method, lewis_fft
from finceptquantlib.pricing import (
    bachelier_implied_vol,
    bachelier_price,
    black76_implied_vol,
    black76_price,
    bs_call,
    bs_implied_vol,
    bs_price,
)
from finceptquantlib.risk.modeling.evt import GEV, GPD, peaks_over_threshold
from finceptquantlib.solver.finance import convexity_adjustment
from finceptquantlib.volatility.local_vol import implied_to_local_vol


# ---------------------------------------------------------------------------
# Special functions and distributions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a,b,x", [(2, 5, 0.1), (2, 5, 0.3), (0.5, 0.5, 0.2), (10, 3, 0.9), (200, 300, 0.4)])
def test_regularized_incomplete_beta(a, b, x):
    assert dist._regularized_incomplete_beta(a, b, x) == pytest.approx(special.betainc(a, b, x), abs=1e-10)


@pytest.mark.parametrize("df", [1, 3, 10, 200, 5000])
@pytest.mark.parametrize("x", [-30.0, -3.0, 0.5, 4.0])
def test_t_cdf_matches_scipy(df, x):
    assert dist.t_cdf(x, df) == pytest.approx(stats.t.cdf(x, df), abs=1e-10)


def test_distribution_classes_match_scipy():
    t = fql.StudentT(5, 1, 2)
    beta = fql.Beta(2, 5)
    f = fql.FDistribution(5, 10)
    for x in (-3.0, 0.0, 2.5):
        assert t.cdf(x) == pytest.approx(stats.t.cdf(x, 5, 1, 2), abs=1e-10)
    for x in (0.1, 0.3, 0.8):
        assert beta.cdf(x) == pytest.approx(stats.beta.cdf(x, 2, 5), abs=1e-10)
    for x in (0.5, 1.0, 3.0):
        assert f.pdf(x) == pytest.approx(stats.f.pdf(x, 5, 10), abs=1e-10)


# ---------------------------------------------------------------------------
# Option pricing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("strike", [60.0, 100.0, 140.0])
@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.8])
@pytest.mark.parametrize("expiry", [0.02, 1.0, 10.0])
def test_implied_vol_round_trips(option_type, strike, sigma, expiry):
    spot, r, q = 100.0, 0.03, 0.01
    for price_fn, iv_fn, vol in (
        (lambda v: bs_price(spot, strike, r, q, v, expiry, option_type),
         lambda p: bs_implied_vol(p, spot, strike, r, q, expiry, option_type), sigma),
        (lambda v: black76_price(spot, strike, r, v, expiry, option_type),
         lambda p: black76_implied_vol(p, spot, strike, r, expiry, option_type), sigma),
        (lambda v: bachelier_price(spot, strike, r, v, expiry, option_type),
         lambda p: bachelier_implied_vol(p, spot, strike, r, expiry, option_type), sigma * spot),
    ):
        target = price_fn(vol)
        assert price_fn(iv_fn(target)) == pytest.approx(target, abs=1e-8)


def test_implied_vol_rejects_arbitrage_price():
    with pytest.raises(ValueError):
        bs_implied_vol(200.0, 100.0, 100.0, 0.03, 0.0, 1.0, OptionType.CALL)


def test_option_type_parsing():
    assert OptionType("CALL") is OptionType.CALL
    assert OptionType(" put ") is OptionType.PUT
    assert OptionType.CALL == "call"
    with pytest.raises(ValueError):
        OptionType("cal")


def _heston_reference(spot, strike, expiry, r, q, v0, kappa, theta, sigma, rho):
    def cf(u):
        d = cmath.sqrt((rho * sigma * 1j * u - kappa) ** 2 + sigma ** 2 * (1j * u + u * u))
        g = (kappa - rho * sigma * 1j * u - d) / (kappa - rho * sigma * 1j * u + d)
        e = cmath.exp(-d * expiry)
        c = (r - q) * 1j * u * expiry + kappa * theta / sigma ** 2 * (
            (kappa - rho * sigma * 1j * u - d) * expiry - 2 * cmath.log((1 - g * e) / (1 - g)))
        dd = (kappa - rho * sigma * 1j * u - d) / sigma ** 2 * (1 - e) / (1 - g * e)
        return cmath.exp(c + dd * v0 + 1j * u * math.log(spot))

    lk = math.log(strike)
    p1 = 0.5 + integrate.quad(
        lambda u: (cmath.exp(-1j * u * lk) * cf(u - 1j) / (1j * u * cf(-1j))).real, 1e-10, 500, limit=1000)[0] / math.pi
    p2 = 0.5 + integrate.quad(
        lambda u: (cmath.exp(-1j * u * lk) * cf(u) / (1j * u)).real, 1e-10, 500, limit=1000)[0] / math.pi
    return spot * math.exp(-q * expiry) * p1 - strike * math.exp(-r * expiry) * p2


@pytest.mark.parametrize("strike,expiry", [(105.0, 1.0), (80.0, 0.1), (130.0, 2.0)])
def test_heston_matches_reference(strike, expiry):
    params = (0.04, 1.5, 0.05, 0.5, -0.7)
    model = HestonModel(HestonParameters(*params), 100.0, 0.03, 0.01)
    ref = _heston_reference(100.0, strike, expiry, 0.03, 0.01, *params)
    assert model.european_call_price(strike, expiry) == pytest.approx(ref, abs=1e-6)
    assert model.european_call_price(strike, expiry, method="fft") == pytest.approx(ref, abs=1e-2)


def test_heston_collapses_to_black_scholes():
    model = HestonModel(HestonParameters(0.0625, 2.0, 0.0625, 1e-4, 0.0), 100.0, 0.03, 0.01)
    assert model.european_call_price(105.0, 1.0) == pytest.approx(
        bs_call(100.0, 105.0, 0.03, 0.01, 0.25, 1.0), abs=1e-6)


def test_fourier_pricers_match_black_scholes():
    spot, strike, expiry, r, q, sigma = 100.0, 110.0, 1.0, 0.05, 0.0, 0.2

    def cf(u):
        drift = math.log(spot) + (r - q - 0.5 * sigma ** 2) * expiry
        return cmath.exp(1j * u * drift - 0.5 * sigma ** 2 * u * u * expiry)

    expected = bs_call(spot, strike, r, q, sigma, expiry)
    assert lewis_fft(cf, spot, strike, r, expiry) == pytest.approx(expected, abs=1e-8)
    assert cos_method(cf, spot, strike, r, expiry) == pytest.approx(expected, abs=1e-8)


def test_flat_implied_vol_gives_flat_local_vol():
    local = implied_to_local_vol(lambda k, t: 0.2, spot=100, rate=0.05, strike=100, expiry=1.0)
    assert local == pytest.approx(0.2, abs=1e-3)


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------

def test_convexity_adjustment_ho_lee_and_hull_white_limit():
    assert convexity_adjustment(0.05, 0.01, 1.0, 1.25) == pytest.approx(0.5 * 0.01 ** 2 * 1.0 * 1.25)
    assert convexity_adjustment(0.05, 0.01, 1.0, 1.25, "hull_white", 1e-6) == pytest.approx(6.25e-5, rel=1e-4)


def test_par_bond_prices_at_par_and_yield_round_trips():
    bond = fql.FixedRateBond(date(2020, 1, 15), date(2030, 1, 15), 0.05, day_count="30/360", settlement_days=0)
    settle = date(2024, 1, 15)
    assert bond.price_from_yield(0.05, settlement_date=settle) == pytest.approx(100.0, abs=1e-10)
    for y in (0.02, 0.08):
        expected = sum(2.5 / (1 + y / 2) ** k for k in range(1, 13)) + 100 / (1 + y / 2) ** 12
        price = bond.price_from_yield(y, settlement_date=settle)
        assert price == pytest.approx(expected, abs=1e-10)
        assert bond.yield_to_maturity(price, settlement_date=settle) == pytest.approx(y, abs=1e-9)


def test_bond_coupons_accrue_on_unadjusted_dates():
    bond = fql.FixedRateBond(date(2020, 1, 15), date(2030, 1, 15), 0.05, day_count="30/360", settlement_days=0)
    coupons = [amount for _, amount in bond.cashflows(date(2024, 1, 15))[:-1]]
    assert coupons == pytest.approx([2.5] * len(coupons))


def test_unknown_compounding_is_rejected():
    bond = fql.FixedRateBond(date(2020, 1, 15), date(2030, 1, 15), 0.05)
    with pytest.raises(ValueError):
        bond.price_from_yield(0.05, settlement_date=date(2024, 1, 15), compounding="weekly")


# ---------------------------------------------------------------------------
# Extreme value theory
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("xi", [-0.2, 0.0, 0.25])
def test_gpd_mle_matches_scipy(xi):
    data = GPD(sigma=0.005, xi=xi).sample(4000, seed=1)
    fitted = GPD.fit_mle(data)
    shape, _, scale = stats.genpareto.fit(data, floc=0)
    assert fitted.xi == pytest.approx(shape, abs=1e-3)
    assert fitted.sigma == pytest.approx(scale, rel=1e-3)


@pytest.mark.parametrize("xi", [-0.2, 0.2])
def test_gev_pwm_recovers_shape(xi):
    fitted = GEV.fit_pwm(GEV(mu=1.0, sigma=0.5, xi=xi).sample(4000, seed=2))
    assert fitted.xi == pytest.approx(xi, abs=0.03)


def test_peaks_over_threshold_var_close_to_true_quantile():
    data = list(stats.t.rvs(4, size=50000, random_state=3))
    result = peaks_over_threshold(data, quantile=0.95)
    assert result.var_estimates[0.99] == pytest.approx(stats.t.ppf(0.99, 4), rel=0.05)
    assert result.es_estimates[0.99] > result.var_estimates[0.99]


# ---------------------------------------------------------------------------
# Calendars
# ---------------------------------------------------------------------------

def test_us_saturday_new_year_observed_on_friday():
    us = fql.USFederalCalendar()
    assert us.is_holiday(date(2021, 12, 31))
    assert us.is_holiday(date(2010, 12, 31))


@pytest.mark.parametrize("day", [date(2020, 5, 8), date(2022, 6, 2), date(2022, 6, 3),
                                 date(2022, 9, 19), date(2023, 5, 8)])
def test_uk_special_bank_holidays(day):
    assert not fql.UKCalendar().is_business_day(day)


@pytest.mark.parametrize("day", [date(2020, 5, 4), date(2022, 5, 30)])
def test_uk_moved_bank_holidays_are_business_days(day):
    assert fql.UKCalendar().is_business_day(day)


# ---------------------------------------------------------------------------
# Library hygiene
# ---------------------------------------------------------------------------

def test_seeded_simulation_does_not_touch_global_rng():
    random.seed(123)
    expected = random.random()
    random.seed(123)
    GPD(sigma=1.0, xi=0.1).sample(10, seed=7)
    assert random.random() == expected


def test_star_import_and_all_are_consistent():
    namespace = {}
    exec("from finceptquantlib import *", namespace)
    assert all(hasattr(fql, name) for name in fql.__all__)
    assert len(fql.__all__) == len(set(fql.__all__))
    assert not [n for n in fql.__all__ if n.startswith("_") and n != "__version__"]
