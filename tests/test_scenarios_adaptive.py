"""Etap 2: błąd standardowy ÊS, adaptacyjna liczba scenariuszy, wspólne liczby losowe, krok modelowy daty."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pf import config
from pf.backtest import engine
from pf.models import dependence as dep
from pf.models.margins import MarginFit
from pf.sim.scenarios import control_portfolios, es_with_se, model_seed, simulate_adaptive

ALPHA = 0.975


def es_normal(alpha):
    return stats.norm.pdf(stats.norm.ppf(alpha)) / (1 - alpha)


def es_t(alpha, nu):
    q = stats.t.ppf(alpha, nu)
    return (nu + q * q) / (nu - 1) * stats.t.pdf(q, nu) / (1 - alpha)


@pytest.mark.parametrize("dist,n", [("normal", 5000), ("t4", 20000)])
def test_es_standard_error_coverage(dist, n):
    """95% przedziały ÊS ± 1,96·SE pokrywają ES w ok. 95% powtórzeń."""
    rng = np.random.default_rng(0)
    true = es_normal(ALPHA) if dist == "normal" else es_t(ALPHA, 4)
    reps, hits = 600, 0
    for _ in range(reps):
        x = rng.standard_normal(n) if dist == "normal" else rng.standard_t(4, n)
        _, es, se = es_with_se(x, ALPHA)
        hits += abs(es - true) <= 1.96 * se
    cover = hits / reps
    assert 0.92 <= cover <= 0.975, cover


def test_es_with_se_basic():
    x = np.arange(1, 41, dtype=float)
    var, es, se = es_with_se(x, ALPHA)
    assert es == pytest.approx(40.0) and var == pytest.approx(39.0) and se > 0
    # SE maleje jak 1/√N
    rng = np.random.default_rng(1)
    se1 = es_with_se(rng.standard_normal(4000), ALPHA)[2]
    se4 = es_with_se(rng.standard_normal(16000), ALPHA)[2]
    assert se4 == pytest.approx(se1 / 2, rel=0.2)


def margin(name, sd=1.0, mu=0.05):
    return MarginFit(name=name, mu=mu, phi=0.0, omega=0.02 * sd ** 2, alpha=0.05, gamma=0.05, beta=0.9, eta=8.0,
                     lam=-0.05, scale=100.0, last_r=0.0, last_eps=0.0, last_sigma2=sd ** 2, loglik=0.0, n_obs=1,
                     converged=True)


SUBFUNDS = {"SF1": {"tickers": ["A", "B"]}, "SF2": {"tickers": ["C", "D"]}}


def test_control_portfolios():
    p = control_portfolios(["A", "B", "C"], SUBFUNDS)
    np.testing.assert_allclose(p["SF1"], [0.5, 0.5, 0.0])
    np.testing.assert_allclose(p["SF2"], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(p["EW"], [0.25, 0.25, 0.5])


def test_adaptive_stopping_and_common_random_numbers():
    names = ["A", "B", "C", "D"]
    fits = [margin(a, sd) for a, sd in zip(names, [1.0, 1.5, 0.6, 0.8])]
    R = np.array([[1, .6, .2, .1], [.6, 1, .2, .1], [.2, .2, 1, .5], [.1, .1, .5, 1.]])
    models = {"gauss": dep.GaussCopula(names, R, n_obs=1), "indep": dep.IndependenceCopula(names, 1, 0.0),
              "student_t": dep.StudentTCopula(names, R, 4.0, n_obs=1)}
    port = control_portfolios(names, SUBFUNDS)
    scfg = {"batch": 500, "n_min": 1000, "n_max": 6000, "rel_se": 0.05}
    t = pd.Timestamp("2020-03-02")
    out = {}
    for name, m in models.items():
        out[name] = simulate_adaptive(fits, m, 5, port, scfg, ALPHA, np.random.default_rng(model_seed(1, t)),
                                      np.random.default_rng(model_seed(1, t, name)), skip=1)
    for s in out.values():
        assert s.n % 500 == 0 and 1000 <= s.n <= 6000
        assert s.converged == (s.stats["rel_se"].max() <= 0.05) or s.n == 6000
        assert set(s.stats["portfolio"]) == {"SF1", "SF2", "EW"}
    # wspólne W: Gauss i niezależność dostały te same wejścia → pierwszy scenariusz aktywa A identyczny
    # (rozkład brzegowy A nie zależy od kopuły, a W_A wchodzi bezpośrednio przez Φ(Φ⁻¹(W)·1))
    n = min(out["gauss"].n, out["indep"].n)
    np.testing.assert_allclose(out["gauss"].R[:n, 0], out["indep"].R[:n, 0])
    # bardzo mała tolerancja → n_max; duża → n_min
    tight = simulate_adaptive(fits, models["gauss"], 5, port, {**scfg, "rel_se": 1e-4}, ALPHA,
                              np.random.default_rng(0), np.random.default_rng(1))
    loose = simulate_adaptive(fits, models["gauss"], 5, port, {**scfg, "rel_se": 0.5}, ALPHA,
                              np.random.default_rng(0), np.random.default_rng(1))
    assert tight.n == 6000 and not tight.converged and loose.n == 1000 and loose.converged
    # prefiks scenariuszy nie zależy od liczby partii
    np.testing.assert_allclose(tight.R[:1000], loose.R)
    # powtarzalność: to samo ziarno → te same scenariusze
    again = simulate_adaptive(fits, models["gauss"], 5, port, {**scfg, "rel_se": 0.5}, ALPHA,
                              np.random.default_rng(0), np.random.default_rng(1))
    np.testing.assert_allclose(again.R, loose.R)


def test_model_seed_stable():
    t = pd.Timestamp("2020-03-02")
    assert model_seed(7, t) == [7, 20200302]
    assert model_seed(7, t, "rvine") == model_seed(7, t, "rvine") != model_seed(7, t, "gauss")


def synthetic_returns(n=420, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    f = rng.standard_t(5, size=(n, 1)) * 0.01
    cols = ["A.WA", "B.WA", "C", "D", "E", "F"]
    X = 0.7 * f + rng.standard_t(5, size=(n, len(cols))) * 0.008
    return pd.DataFrame(X, index=idx, columns=cols)


def test_engine_model_step_with_cache(tmp_path):
    cfg = config.load("models")
    cfg = {**cfg, "rebalance": {**cfg["rebalance"], "window_days": 350, "min_obs": 300}}
    bcfg = {**config.load("backtest"), "execution_lag": 1,
            "scenarios": {"batch": 200, "n_min": 400, "n_max": 800, "rel_se": 0.05, "rel_se_last": 0.03}}
    subfunds = {"SF1": {"tickers": ["A.WA", "B.WA", "C"]}, "SF2": {"tickers": ["D", "E", "F"]}}
    r = synthetic_returns()
    t = r.index[380]
    dm = engine.fit_margins_step(t, r, cfg, jobs=1, cache_dir=tmp_path)
    assert dm.window_end < t and not dm.cached["margins"] and dm.U.shape[1] == 6
    engine.fit_models_step(dm, cfg, ["rvine", "gauss", "empirical"])
    assert not any(dm.cached[m] for m in ("rvine", "gauss", "empirical"))
    scen = engine.scenarios_step(dm, cfg, bcfg, subfunds)
    table = engine.scenario_stats_table(t, scen)
    assert set(table["model"]) == {"rvine", "gauss", "empirical"} and (table["n"] >= 400).all()
    # drugi przebieg: modele z pamięci podręcznej, identyczne scenariusze
    dm2 = engine.fit_models_step(engine.fit_margins_step(t, r, cfg, jobs=1, cache_dir=tmp_path), cfg,
                                 ["rvine", "gauss", "empirical"])
    assert dm2.cached["margins"] and all(dm2.cached[m] for m in ("rvine", "gauss", "empirical"))
    scen2 = engine.scenarios_step(dm2, cfg, bcfg, subfunds)
    for m in scen:
        np.testing.assert_allclose(scen2[m].R, scen[m].R)
    # zmiana danych po t−1 nie zmienia modeli ani scenariuszy daty t
    r3 = r.copy()
    r3.loc[r3.index >= t] *= 3.0
    dm3 = engine.fit_models_step(engine.fit_margins_step(t, r3, cfg, jobs=1, cache_dir=None), cfg, ["gauss"])
    np.testing.assert_allclose(engine.scenarios_step(dm3, cfg, bcfg, subfunds)["gauss"].R, scen["gauss"].R)
    # zmiana danych w oknie → nowy klucz pamięci podręcznej
    r4 = r.copy()
    r4.iloc[100, 0] += 0.01
    assert not engine.fit_margins_step(t, r4, cfg, jobs=1, cache_dir=tmp_path).cached["margins"]
