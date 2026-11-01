"""Etap 4: testy VaR/ES na ciągach o znanych własnościach, realizacja stóp, miary, test dymny backtestu."""

import time

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pf import config
from pf.backtest import engine, metrics, risk_tests
from pf.backtest.realized import pit, realize
from pipelines import backtest as bt

ALPHA = 0.975
P = 1 - ALPHA


# --------------------------------------------------------------------------- testy VaR/ES

def test_kupiec_and_binomial():
    rng = np.random.default_rng(0)
    ok = [risk_tests.kupiec(rng.random(1000) < P, P)["p_uc"] for _ in range(200)]
    assert 0.02 < np.mean(np.array(ok) < 0.05) < 0.10              # rozmiar ≈ 5%
    bad = risk_tests.kupiec(rng.random(1000) < 3 * P, P)
    assert bad["p_uc"] < 1e-4 and bad["p_binom"] < 1e-4
    zero = risk_tests.kupiec(np.zeros(69, bool), P)
    assert zero["exceedances"] == 0 and zero["expected"] == pytest.approx(69 * P)
    assert zero["lr_uc"] == pytest.approx(-2 * 69 * np.log(1 - P))


def test_christoffersen_independence():
    rng = np.random.default_rng(1)
    iid = risk_tests.christoffersen(rng.random(2000) < 0.05, 0.05)
    assert iid["p_ind"] > 0.01
    h = np.zeros(2000, bool)
    state = False
    for i in range(2000):                                          # łańcuch Markowa z silną persystencją
        state = rng.random() < (0.6 if state else 0.03)
        h[i] = state
    clus = risk_tests.christoffersen(h, 0.05)
    assert clus["p_ind"] < 1e-6 and clus["n11"] > 0 and clus["p_cc"] < 1e-6


def _forecast_sample(T, scale, seed):
    """Prognozy N(0,1)-scenariuszowe; realizacje z N(0, scale²)."""
    rng = np.random.default_rng(seed)
    N = 4000
    Z = rng.standard_normal(N)
    tail = -np.sort(-Z)[: int(np.ceil(P * N))]
    var, es = np.sort(Z)[::-1][len(tail)], tail.mean()
    loss = scale * rng.standard_normal(T)
    u = np.array([np.mean(Z <= x) for x in loss])
    return loss, np.full(T, var), np.full(T, es), u, [tail] * T


def test_z2_and_du_escanciano_size_and_power():
    loss, var, es, u, tails = _forecast_sample(600, 1.0, 2)
    h0 = risk_tests.backtest_series(loss, var, es, u, tails, ALPHA, 5000, (1, 5), seed=0)
    assert h0["p_z2"] > 0.05 and h0["p_de_uc"] > 0.05 and abs(h0["z2_sim_mean"]) < 0.05
    loss, var, es, u, tails = _forecast_sample(600, 1.6, 3)        # ryzyko niedoszacowane
    h1 = risk_tests.backtest_series(loss, var, es, u, tails, ALPHA, 5000, (1, 5), seed=0)
    assert h1["z2"] < 0 and h1["p_z2"] < 0.01 and h1["p_de_uc"] < 0.01 and h1["p_uc"] < 0.01
    # autokorelacja PIT → warunkowy test Du–Escanciano odrzuca
    rng = np.random.default_rng(4)
    e = np.zeros(3000)
    for i in range(1, 3000):
        e[i] = 0.9 * e[i - 1] + np.sqrt(1 - 0.81) * rng.standard_normal()
    de = risk_tests.du_escanciano(stats.norm.cdf(e), ALPHA, (1, 5))
    assert de["p_de_cc1"] < 0.01
    iid = risk_tests.du_escanciano(rng.random(3000), ALPHA, (1, 5))
    assert iid["p_de_cc5"] > 0.01 and iid["p_de_uc"] > 0.01


def test_pit_from_tail_and_grid():
    losses = np.random.default_rng(5).standard_normal(4000)
    grid = np.arange(0.005, 1.0, 0.005)
    tail = -np.sort(-losses)[:100]
    q = np.quantile(losses, grid)
    for L in (-1.0, 0.0, 1.5, 2.3, 3.0, 10.0):
        exact = np.mean(losses <= L)
        assert pit(L, 4000, tail, q, grid) == pytest.approx(exact, abs=0.003 if L < tail.min() else 1e-12)


# --------------------------------------------------------------------------- realizacja i miary

def test_realize_drift_turnover_and_costs():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    r = pd.DataFrame({"A": 0.0, "B": 0.0}, index=idx)
    r.loc["2020-01-15", "A"] = np.log1p(0.10)
    dates = pd.DatetimeIndex(["2020-01-02", "2020-02-03", "2020-03-02"])
    w = pd.DataFrame({"date": np.repeat(dates[:2], 2), "sid": "s", "ticker": ["A", "B"] * 2, "weight": 0.5})
    daily, per = realize(r, w, pd.Series({"s": 1}), dates, 0.001, None)
    assert per["r_gross"].tolist() == pytest.approx([0.05, 0.0])
    # okres 2: wagi docelowe 50/50 po dryfie 0,55/0,45 → Σ|Δx| = 0,0952…
    d = np.array([0.55, 0.5]) / 1.05
    assert per["traded"].iloc[1] == pytest.approx(np.abs(d - 0.5).sum())
    assert per["cost"].tolist() == pytest.approx([0.001, 0.001 * np.abs(d - 0.5).sum()])
    assert per["r_net"].iloc[0] == pytest.approx(0.999 * 1.05 - 1)
    g = daily[daily["date"] == dates[0]]
    assert np.prod(1 + g["r_net"].astype(float)) - 1 == pytest.approx(per["r_net"].iloc[0], abs=1e-6)


def test_metrics_basic():
    r = np.array([0.02, -0.01, 0.03, -0.05, 0.01, 0.02])
    rf = np.full(6, 0.001)
    assert metrics.max_drawdown(r) == pytest.approx(0.95 - 1)
    assert metrics.realized_ce(r, rf, 5.0) < metrics.realized_ce(r, rf, 1.0) < np.mean((1 + r) / (1 + rf) - 1)
    assert metrics.realized_ce(np.full(6, 0.01), np.zeros(6), 7.0) == pytest.approx(0.01)
    sr, se = metrics.sharpe(r, rf)
    x = r - rf
    assert sr == pytest.approx(x.mean() / x.std(ddof=1) * np.sqrt(12))
    assert metrics.omega(r) == pytest.approx(0.08 / 0.06)


# --------------------------------------------------------------------------- test dymny end-to-end

def synthetic_panel(n=400, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    cols = ["P1.WA", "P2.WA", "P3.WA", "U1", "U2", "U3", "V1", "V2", "V3"]
    f = rng.standard_t(5, size=(n, 1)) * 0.008
    vol = np.array([1.2, 1.0, 0.9, 1.0, 0.8, 1.1, 0.4, 0.3, 0.5]) * 0.01
    X = 0.6 * f + rng.standard_t(5, size=(n, 9)) * vol + 0.0002
    return pd.DataFrame(X, index=idx, columns=cols)


def smoke_configs():
    cfg = config.load("models")
    cfg = {**cfg, "rebalance": {**cfg["rebalance"], "window_days": 250, "min_obs": 200},
           "dependence": {**cfg["dependence"], "models": ["rvine", "gauss"]}}
    bcfg = config.load("backtest")
    bcfg = {**bcfg, "calibration": ["2019-01-01", "2020-01-31"], "holdout": ["2020-02-01", "2020-12-31"],
            "grid": {**bcfg["grid"], "models": ["rvine", "gauss"]},
            "scenarios": {"batch": 300, "n_min": 600, "n_max": 900, "rel_se": 0.05, "rel_se_last": 0.05},
            "risk_tests": {**bcfg["risk_tests"], "z2_simulations": 500}}
    sf = {"SF1": {"name": "PL", "tickers": ["P1.WA", "P2.WA", "P3.WA"]},
          "SF2": {"name": "US", "tickers": ["U1", "U2", "U3"]}, "SF3": {"name": "Obl", "tickers": ["V1", "V2", "V3"]}}
    return cfg, bcfg, config.load("profiles"), sf


def test_backtest_smoke_end_to_end(tmp_path):
    t_start = time.time()
    cfg, bcfg, pcfg, sf = smoke_configs()
    r = synthetic_panel()
    dates = bt.all_dates(r, cfg, bcfg, sf)[:3]
    assert len(dates) == 3
    for i, t in enumerate(dates):
        nxt = dates[i + 1] if i + 1 < len(dates) else None
        s = engine.run_date(t, nxt, r, cfg, bcfg, pcfg, sf, tmp_path, tmp_path / "cache", latest=nxt is None)
        assert s["n_assets"] == 9
    rf = pd.Series(0.03, index=r.index)
    out = bt.consolidate(tmp_path, r, cfg, bcfg, pcfg, sf, rf=rf)
    per = out["periods"]
    base = per[per["variant"] == "base"]
    assert base["sid"].nunique() == 2 * 4 * 5 + 1 + 5                # siatka: 2 modele × 4 strategie × 5 profili + ew + 5 mv
    assert (per["r_net"] <= per["r_gross"] + 1e-12).all() and (per["cost"] > 0).all()
    assert set(per["variant"]) > {"base", "lag0", "q10", "rhoS"}
    assert per["date"].nunique() == 2                                # ostatnia data bez okresu realizacji
    fc = out["forecasts"]
    ok = fc["loss"].notna()
    assert ((fc.loc[ok, "pit"] >= 0) & (fc.loc[ok, "pit"] <= 1)).all() and (fc["es"] >= fc["var"] - 1e-12).all()
    assert len(out["tests"]) > 0 and {"p_uc", "p_ind", "p_z2", "p_de_uc"} <= set(out["tests"].columns)
    assert set(out["logscore"]["model"]) == {"rvine", "gauss"}
    assert {"rvine", "gauss", "empirical"} >= set(out["gof"]["model"])
    sf_per = out["periods_sf"]
    assert set(sf_per["subfund"]) == {"SF1", "SF2", "SF3"}
    tw = pd.read_parquet(tmp_path / "weights_tickers.parquet")
    sums = tw.groupby(["date", "model", "strategy", "profile"], observed=True)["x"].sum()
    assert np.allclose(sums, 1.0, atol=1e-6) and np.allclose(tw["x"], tw["w_sf"] * tw["W_sf"], atol=1e-7)
    assert tw.groupby(["model", "strategy", "profile"], observed=True).ngroups == 2 * 4 * 5 + 1 + 5
    # limity profili w wagach look-through
    W = out["weights_W"]
    assert (W.groupby(["date", "sid"])["weight"].sum() - 1).abs().max() < 1e-9
    assert time.time() - t_start < 60


def test_no_look_ahead(tmp_path):
    cfg, bcfg, pcfg, sf = smoke_configs()
    bcfg = {**bcfg, "sensitivity": {**bcfg["sensitivity"], "enabled": False},
            "grid": {**bcfg["grid"], "models": ["gauss"]}}
    r = synthetic_panel()
    t = bt.all_dates(r, cfg, bcfg, sf)[1]
    engine.run_date(t, None, r, cfg, bcfg, pcfg, sf, tmp_path / "a", None)
    r2 = r.copy()
    r2.loc[r2.index >= t] = r2.loc[r2.index >= t] * -5.0 + 0.01     # inne dane od t (po t−1)
    engine.run_date(t, None, r2, cfg, bcfg, pcfg, sf, tmp_path / "b", None)
    tag = t.strftime("%Y-%m-%d")
    wa = pd.read_parquet(tmp_path / "a" / "dates" / tag / "weights.parquet")
    wb = pd.read_parquet(tmp_path / "b" / "dates" / tag / "weights.parquet")
    pd.testing.assert_frame_equal(wa, wb)
