"""Nakładka opcyjna – Etap 2: prognoza GARCH zmienności, m_t, σ z indeksów Cboe, proxy, korekta skośności."""

import numpy as np
import pandas as pd
import pytest
from scipy import integrate

from pf.models.margins import MarginFit, simulate_daily, skewt_ppf
from pf.options import implied_vol as ivm

ICFG = {"default_source": "proxy", "sources": {"GLD": "GVZ"}, "vix_tenors_days": {"VIX9D": 9, "VIX": 30},
        "atm_adj": 1.0, "sigma_min": 0.01, "sigma_max": 1.5}


def fit(**kw) -> MarginFit:
    base = dict(name="X", mu=0.02, phi=0.05, omega=0.03, alpha=0.04, gamma=0.08, beta=0.88, eta=6.0, lam=-0.15,
                scale=100.0, last_r=-1.2, last_eps=-1.5, last_sigma2=2.0, loglik=0.0, n_obs=1000, converged=True,
                eps_absmax=10.0)
    base.update(kw)
    return MarginFit(**base)


@pytest.mark.parametrize("eta,lam", [(5.0, 0.0), (4.0, -0.3), (8.0, 0.25), (2.6, -0.1)])
def test_skewt_density_moments(eta, lam):
    f = lambda z: float(ivm.skewt_pdf(z, eta, lam))  # noqa: E731
    total = integrate.quad(f, -np.inf, np.inf, limit=200)[0]
    mean = integrate.quad(lambda z: z * f(z), -np.inf, np.inf, limit=200)[0]
    assert total == pytest.approx(1.0, abs=1e-7) and mean == pytest.approx(0.0, abs=1e-6)
    if eta > 3:
        var = integrate.quad(lambda z: z * z * f(z), -np.inf, np.inf, limit=200)[0]
        assert var == pytest.approx(1.0, abs=1e-5)


def test_neg_second_moment_symmetric_and_vs_monte_carlo():
    assert ivm.neg_second_moment(7.0, 0.0) == pytest.approx(0.5, abs=1e-8)
    u = (np.arange(400_000) + 0.5) / 400_000
    z = skewt_ppf(u, 6.0, -0.3)
    mc = np.mean(z ** 2 * (z < 0))
    assert ivm.neg_second_moment(6.0, -0.3) == pytest.approx(mc, rel=5e-3)
    assert ivm.neg_second_moment(6.0, -0.3) > 0.5 > ivm.neg_second_moment(6.0, 0.3)


def test_variance_path_matches_simulation():
    f = fit()
    rng = np.random.default_rng(3)
    S, H = 200_000, 8
    r = simulate_daily([f], rng.random((S, H, 1)), eps_cap=None)[:, :, 0] * f.scale
    prev = np.concatenate([np.full((S, 1), f.last_r), r[:, :-1]], axis=1)
    eps = r - f.mu - f.phi * prev
    sim = (eps ** 2).mean(axis=0)
    ana = ivm.variance_path(f.next_sigma2(), f.omega, ivm.persistence(f), H)
    np.testing.assert_allclose(sim, ana, rtol=0.03)
    # opóźnienie L: kroki L+1, …, L+n
    np.testing.assert_allclose(ivm.variance_path(f.next_sigma2(), f.omega, ivm.persistence(f), 5, skip=3), ana[3:8])


def test_garch_vol_constant_variance_and_deterministic():
    f = fit(alpha=0.0, gamma=0.0, beta=0.0, omega=1.44)
    assert ivm.garch_vol(f, 21, 1) == pytest.approx(np.sqrt(252 * 1.44) / 100)
    det = fit(kind="deterministic")
    assert ivm.garch_vol(det, 21) == 0.0


def test_vrp_multiplier():
    assert ivm.vrp_multiplier(20.0, 0.16) == pytest.approx(1.25)
    assert np.isnan(ivm.vrp_multiplier(np.nan, 0.16)) and np.isnan(ivm.vrp_multiplier(20.0, 0.0))


def test_index_vol_vix_interpolation():
    lv = {"VIX9D": 16.0, "VIX": 20.0}
    tenors = ICFG["vix_tenors_days"]
    assert ivm.index_vol("VIX", lv, 30, tenors) == pytest.approx(0.20)
    assert ivm.index_vol("VIX", lv, 9, tenors) == pytest.approx(0.16)
    w9, w30 = 0.16 ** 2 * 9 / 365, 0.20 ** 2 * 30 / 365
    w20 = w9 + (w30 - w9) * (20 - 9) / (30 - 9)
    assert ivm.index_vol("VIX", lv, 20, tenors) == pytest.approx(np.sqrt(w20 / (20 / 365)))
    assert ivm.index_vol("VIX", lv, 33, tenors) == pytest.approx(0.20)             # płaska ekstrapolacja
    assert ivm.index_vol("VIX", {"VIX": 20.0}, 20, tenors) == pytest.approx(0.20)  # brak VIX9D (przed 2011)
    assert ivm.index_vol("GVZ", {"GVZ": 15.0}, 28, tenors, atm_adj=0.9) == pytest.approx(0.135)


def test_sigma_table_sources_fallback_and_shift():
    fits = {"GLD": fit(name="GLD"), "AAA.WA": fit(name="AAA.WA")}
    sh = ivm.garch_vol(fits["AAA.WA"], 21, 1)
    tab = ivm.sigma_table(["GLD", "AAA.WA"], fits, 21, 1, 1.2, {"GVZ": 15.0, "VIX": 20.0}, 30, ICFG)
    assert tab.loc["GLD", "iv_source"] == "GVZ" and tab.loc["GLD", "sigma_imp"] == pytest.approx(0.15)
    assert tab.loc["AAA.WA", "sigma_imp"] == pytest.approx(1.2 * sh)
    missing = ivm.sigma_table(["GLD"], fits, 21, 1, 1.2, {"VIX": 20.0}, 30, ICFG)     # GVZ przed 2009-09
    assert missing.loc["GLD", "iv_source"] == "proxy" and missing.loc["GLD", "iv_source_cfg"] == "GVZ"
    shifted = ivm.sigma_table(["GLD", "AAA.WA"], fits, 21, 1, 1.2, {"GVZ": 15.0}, 30, ICFG, shift_pp=3.0)
    assert shifted.loc["GLD", "sigma_imp"] == pytest.approx(0.15)                   # indeks – bez przesunięcia
    assert shifted.loc["AAA.WA", "sigma_imp"] == pytest.approx(1.2 * sh + 0.03)
    low = ivm.sigma_table(["AAA.WA"], fits, 21, 1, 1e-4, {}, 30, ICFG)
    assert low.loc["AAA.WA", "sigma_imp"] == ICFG["sigma_min"]


def test_strike_vol_skew():
    s = ivm.strike_vol(0.2, [95.0, 100.0, 105.0], 100.0, beta=-0.5)
    np.testing.assert_allclose(s, 0.2 - 0.5 * np.log([0.95, 1.0, 1.05]))
    np.testing.assert_allclose(ivm.strike_vol(0.2, [95.0, 105.0], 100.0), 0.2)


def test_market_multiplier_atm_adjustment():
    assert ivm.market_multiplier(20.0, 0.16, {"atm_adj": 1.0}) == pytest.approx(1.25)
    assert ivm.market_multiplier(20.0, 0.16, {"atm_adj": 0.8}) == pytest.approx(1.0)
    assert ivm.market_multiplier(20.0, 0.16, {"atm_adj": 0.8, "atm_adj_in_m": False}) == pytest.approx(1.25)


# --------------------------------------------------------------------------- ograniczenie m_t (m_cap)

def _spx_returns(n=400, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-05", periods=n)
    return pd.Series(rng.standard_t(6, size=n) * 0.009, index=idx)


def test_multiplier_history_constant_variance():
    r = _spx_returns()
    vix = pd.Series(20.0, index=r.index)
    f = fit(mu=0.0, phi=0.0, alpha=0.0, gamma=0.0, beta=0.0, omega=1.44)
    h = ivm.multiplier_history(f, r, vix, 21, skip=1)
    assert h.index.equals(r.index[2:]) and len(h) == len(r) - 2
    np.testing.assert_allclose(h.to_numpy(), 0.20 / (np.sqrt(252 * 1.44) / 100), rtol=1e-12)
    half = ivm.multiplier_history(f, r, vix, 21, skip=1, atm_adj=0.5)
    np.testing.assert_allclose(half.to_numpy(), 0.5 * h.to_numpy(), rtol=1e-12)


def test_multiplier_history_matches_m_t_definition_and_window_only():
    r = _spx_returns()
    vix = pd.Series(15.0 + 10.0 * np.abs(np.sin(np.arange(len(r)) / 17.0)), index=r.index)
    f = fit()
    h = ivm.multiplier_history(f, r, vix, 21, skip=1)
    # ostatni element: model ze stanem w przedostatniej sesji okna i VIX z ostatniej sesji okna (jak m_t w dacie t)
    eps, s2 = ivm.filtered_variance(f, r.to_numpy())
    state = fit(last_eps=float(eps[-2]), last_sigma2=float(s2[-2]))
    m_last = ivm.vrp_multiplier(float(vix.iloc[-1]), ivm.garch_vol(state, 21, 1))
    assert h.iloc[-1] == pytest.approx(m_last, rel=1e-12)
    # historia do sesji s zależy od danych po s tylko przez start filtru (wariancja próbkowa okna), który zanika z β
    cut = 300
    h_cut = ivm.multiplier_history(f, r.iloc[:cut], vix.iloc[:cut], 21, skip=1)
    np.testing.assert_allclose(h_cut.to_numpy()[150:], h.iloc[150:cut - 2].to_numpy(), rtol=1e-6)
    # brak VIX w sesji → NaN w historii, bez wpływu na pozostałe sesje
    gap = vix.copy()
    gap.iloc[50] = np.nan
    hg = ivm.multiplier_history(f, r, gap, 21, skip=1)
    assert np.isnan(hg.iloc[48]) and hg.drop(hg.index[48]).equals(h.drop(h.index[48]))


def test_multiplier_cap_quantile_min_obs_and_disabled():
    hist = np.concatenate([np.linspace(0.5, 1.5, 300), [np.nan, np.inf]])
    cap = ivm.multiplier_cap(hist, {"quantile": 0.95, "min_obs": 250})
    assert cap == pytest.approx(np.quantile(np.linspace(0.5, 1.5, 300), 0.95))
    assert ivm.multiplier_cap(hist, {"quantile": 0.95, "min_obs": 301}) == float("inf")
    assert ivm.multiplier_cap(hist, {"quantile": None}) == float("inf")
    assert ivm.multiplier_cap(hist, None) == float("inf")
    assert ivm.capped_multiplier(2.5, cap) == pytest.approx(cap)
    assert ivm.capped_multiplier(1.0, cap) == 1.0 and ivm.capped_multiplier(1.0, float("inf")) == 1.0
    assert np.isnan(ivm.capped_multiplier(float("nan"), cap))
