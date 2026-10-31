"""Etap 6: MCS, test różnicy Sharpe'a (Ledoit–Wolf), wybór κ z ograniczeniem monotoniczności, tabele rankingu."""

import itertools

import numpy as np
import pandas as pd
import pytest

from pf.backtest.ranking import add_ranks, crra_loss, mcs, monotone_kappa, series_metrics, sharpe_diff_test
from pf.opt.profiles import Profile
from pipelines import ranking as rk


def test_sharpe_diff_size_and_power():
    rng = np.random.default_rng(0)
    p0 = []
    for i in range(40):
        z = rng.standard_normal((120, 2)) * 0.04 + 0.006
        p0.append(sharpe_diff_test(z[:, 0], z[:, 1], 3, 300, seed=i)["p_sharpe_vs_ew"])
    assert 0.0 <= np.mean(np.array(p0) < 0.05) <= 0.2
    x = rng.standard_normal(300) * 0.04 + 0.02
    y = 0.7 * x + rng.standard_normal(300) * 0.03 - 0.005
    res = sharpe_diff_test(x, y, 3, 999, seed=1)
    assert res["sr_diff_m"] > 0 and res["p_sharpe_vs_ew"] < 0.01


def test_mcs_excludes_bad_models():
    rng = np.random.default_rng(2)
    L = pd.DataFrame(rng.standard_normal((200, 4)) + np.array([0, 0.05, 1.0, 1.5]), columns=list("abcd"))
    m = mcs(L, 0.10, 1000, seed=3)
    assert m.loc["a", "in_mcs"] and not m.loc["c", "in_mcs"] and not m.loc["d", "in_mcs"]


def test_crra_loss_orders_like_ce():
    r1, r2 = np.full(24, 0.01), np.tile([0.05, -0.03], 12)
    rf = np.zeros(24)
    assert crra_loss(r1, rf, 5).mean() < crra_loss(r2, rf, 5).mean()


def test_monotone_kappa_matches_brute_force():
    rng = np.random.default_rng(4)
    ks = [0.0, 0.15, 0.35, 1.0]
    order = ["P1", "P2", "P3"]
    ce = pd.DataFrame(rng.normal(size=(3, 4)), index=order, columns=ks)
    sel = monotone_kappa(ce, order)
    best, best_v = None, -np.inf
    for combo in itertools.product(ks, repeat=3):
        if all(a >= b for a, b in zip(combo[:-1], combo[1:])):
            v = sum(ce.loc[p, k] for p, k in zip(order, combo))
            if v > best_v:
                best, best_v = combo, v
    assert tuple(sel[p] for p in order) == best


def fake_periods(T=72, seed=5):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=T, freq="MS")
    rows = []
    sids = [("rvine", "ce_dep", "P1"), ("gauss", "ce_dep", "P1"), ("rvine", "ce", "P1"), ("-", "mv_hist", "P1"),
            ("-", "ew", "ALL")]
    for j, (m, s, p) in enumerate(sids):
        r = rng.standard_normal(T) * 0.03 + 0.004 + 0.001 * j
        for t, d in enumerate(dates):
            rows.append({"sid": f"{m}|{s}|{p}|base", "date": d, "end": d + pd.offsets.MonthBegin(1), "r_net": r[t],
                         "r_gross": r[t] + 1e-4, "rf": 0.002, "sample": "calibration" if t < 48 else "holdout",
                         "model": m, "strategy": s, "profile": p, "variant": "base", "turnover": 0.1, "cost": 1e-4})
    return pd.DataFrame(rows)


def test_rank_tables_and_markdown():
    per = fake_periods()
    gam = {"P1": 10.0}
    met = series_metrics(per[per["sample"] == "holdout"], gam, 0.975)
    assert set(met["eval_profile"]) == {"P1"} and len(met) == 5
    ranked = add_ranks(met, {"ce_ann": 3, "sharpe": 1})
    assert ranked["rank_composite"].between(1, 5).all()
    rcfg = {"mcs_size": 0.1, "mcs_reps": 200, "sharpe_test": {"block": 3, "reps": 99, "benchmark": "-|ew|ALL|base"},
            "composite_weights": {"ce_ann": 3, "sharpe": 1, "max_dd": 1}, "test_level": 0.05}
    tests = pd.DataFrame({"sid": per["sid"].unique(), "sample": "holdout", "p_uc": 0.5, "p_cc": 0.4, "p_z2": 0.3,
                          "p_de_uc": 0.2, "p_de_cc5": 0.6, "exceedances": 1, "expected": 0.6, "T": 24,
                          "model": [s.split("|")[0] for s in per["sid"].unique()]})
    tab = rk.rank_sample(per, tests, gam, rcfg, 0.975, None, "holdout")
    assert tab["ce_ann"].is_monotonic_decreasing and "in_mcs" in tab and "p_sharpe_vs_ew" in tab
    prof = {"P1": Profile("P1", "konserwatywny", 10.0, 1.0, 0.2, 0.05, 0.4)}
    md = rk.ranking_markdown(tab, "holdout", prof, rcfg, 24)
    assert "## P1" in md and "rvine/ce_dep" in md
