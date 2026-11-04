"""Nakładka opcyjna – Etap 5: ograniczenia grupowe poziomu 2, kolumny SFk_OV, ten sam budżet ES, MC roczne."""

import numpy as np
import pandas as pd
import pytest

from pf import config
from pf.backtest.data import rebalance_dates
from pf.opt.cvar import empirical_es, max_return_es
from pf.opt.profiles import SubfundData, load_profiles, solve_ce
from pf.options import calendar, engine, paths, portfolio
from tests.options_synth import SUBFUNDS, models_config, overlay_config, synthetic_market

PROFILES = load_profiles(config.load("profiles"))


def test_solve_ce_and_lp_respect_group_constraints():
    rng = np.random.default_rng(1)
    R = rng.normal([0.01, 0.012, 0.008, 0.011], [0.05, 0.06, 0.03, 0.06], size=(4000, 4))
    A, b = np.array([[1.0, 0.0, 0.0, 1.0]]), np.array([0.3])
    w, info = solve_ce(R, 3.0, 0.0, 0.8, A_ub=A, b_ub=b, w0=np.array([0.0, 0.5, 0.5, 0.0]))
    assert abs(w.sum() - 1) < 1e-9 and w[0] + w[3] <= 0.3 + 1e-7
    w_free, _ = solve_ce(R, 3.0, 0.0, 0.8)
    assert w_free[0] + w_free[3] > 0.3                              # ograniczenie jest wiążące
    sol = max_return_es(R, 0.975, 0.2, 0.8, A, b)
    assert sol.feasible and sol.weights[0] + sol.weights[3] <= 0.3 + 1e-9


def _sfs(S=3000, seed=2):
    rng = np.random.default_rng(seed)
    z = rng.standard_t(5, size=(S, 4)) * 0.04
    g = np.exp(0.004 + z)
    f = np.ones_like(g)
    R = g - 1
    sfs = [SubfundData("SF1", ["A.WA", "B.WA"], R[:, :2], float(R[:, :2].mean(axis=1).std())),
           SubfundData("SF2", ["C.WA", "D.WA"], R[:, 2:], float(R[:, 2:].mean(axis=1).std()))]
    gross = {t: (g[:, k], f[:, k]) for k, t in enumerate(["A.WA", "B.WA", "C.WA", "D.WA"])}
    return sfs, gross


def _puts(tickers, h=1.0, prem=0.01):
    return pd.DataFrame({"ticker": tickers, "type": "put", "moneyness": 0.975, "h": h, "premium_rel": prem,
                         "delta": -0.4})


@pytest.mark.parametrize("strategy", ["ce_dep", "maxret_es", "min_es"])
def test_optimize_with_overlay_constraints_and_budget(strategy):
    sfs, gross = _sfs()
    prof = PROFILES["P4"]
    pos = {"SF1": _puts(["A.WA", "B.WA"])}
    out = portfolio.optimize_with_overlay(sfs, gross, pos, prof, strategy, 0.975, {}, 0.03)
    res, R, names = out["ov"]
    assert names == ["SF1", "SF2", "SF1_OV"]
    W = res.W.to_numpy()
    assert W.sum() == pytest.approx(1.0) and (W >= -1e-12).all()
    assert W[0] + W[2] <= prof.w_max + 1e-7 and W[1] <= prof.w_max + 1e-7
    if strategy == "maxret_es":                                    # ten sam budżet ES co wariant bazowy
        limit = empirical_es(-out["base"][1] @ np.full(2, 0.5), 0.975)
        assert empirical_es(-R @ W, 0.975) <= limit + 1e-8
    # h ≡ 0: kolumna SFk_OV = kolumna SFk
    zero = portfolio.optimize_with_overlay(sfs, gross, {"SF1": _puts(["A.WA", "B.WA"], h=0.0)}, prof, strategy,
                                           0.975, {}, 0.03)
    np.testing.assert_allclose(zero["ov"][1][:, 2], zero["ov"][1][:, 0], atol=1e-14)


def test_es_scale_matches_budget():
    rng = np.random.default_rng(3)
    r = rng.normal(0.01, 0.06, 20000)
    rf = 0.004
    es, target = empirical_es(-r, 0.975), 0.06
    a = portfolio.es_scale(es, target, rf)
    assert 0 < a < 1
    assert empirical_es(-(a * r + (1 - a) * rf), 0.975) == pytest.approx(target, rel=1e-10)
    assert portfolio.es_scale(0.05, 0.06, rf) == 1.0
    assert portfolio.ov_allowed("P3", overlay_config()) and not portfolio.ov_allowed("P2", overlay_config())


@pytest.fixture(scope="module")
def synth():
    market = synthetic_market()
    cfg, ocfg = models_config(), overlay_config()
    dates = rebalance_dates(market.index, cfg["rebalance"]["window_days"])
    periods = calendar.overlay_periods(market.index, dates, lag=1)
    return market, cfg, ocfg, periods


def test_portfolio_step_and_realize(synth):
    market, cfg, ocfg, periods = synth
    pcfg = config.load("profiles")
    W, L1, XA, pos_by_date = [], [], [], {}
    for j in range(3):
        per = periods.iloc[j].to_dict()
        ds = engine.date_step(per, market, cfg, ocfg, SUBFUNDS, seed=5)
        pos = engine.positions_with_premia(ds["decisions"], ds["ledger"], "P3")
        pos_by_date[per["date"]] = pos
        out = engine.portfolio_step(per, market, ds["model"], cfg, ocfg, pcfg, SUBFUNDS, pos, seed=5)
        W.append(out["weights"]), L1.append(out["level1"]), XA.append(out["exante"])
    W, L1, XA = (pd.concat(x, ignore_index=True) for x in (W, L1, XA))
    assert set(XA["variant"]) == {"base", "ov"} and set(XA["profile"]) == {"P3", "P4", "P5"}
    assert np.allclose(W.groupby(["date", "profile", "strategy", "variant"])["W"].sum(), 1.0)
    assert "SF10" in set(W["column"])                               # SF10 w portfelu, bez nakładki
    assert not W["column"].str.startswith("SF10_").any()
    real = engine.portfolio_realize(periods.iloc[:3], market, W, L1, XA, pos_by_date, ocfg, 0.001, 1)
    assert len(real) == 3 * 3 * 2 * 2 and real[["r_gross", "r_net", "r_es_matched"]].notna().all().all()
    assert (real["es_scale"] <= 1).all() and (real["cost"] >= 0).all()
    # ten sam ÊS po zmieszaniu z lokatą: min(ES_base, ES_ov) w obu wariantach
    one = real[(real["profile"] == "P4") & (real["strategy"] == "ce_dep")]
    for d, g in one.groupby("date"):
        es, a, rf = g["es_exante"].to_numpy(), g["es_scale"].to_numpy(), g["rf"].iloc[0]
        np.testing.assert_allclose(a * es - (1 - a) * rf, es.min(), rtol=1e-9)


def test_annual_paths_policies(synth):
    market, cfg, ocfg, periods = synth
    per = periods.iloc[3].to_dict()
    ds = engine.date_step(per, market, cfg, ocfg, SUBFUNDS, seed=5)
    model = ds["model"]
    sf = engine.overlay_subfunds(SUBFUNDS, ocfg)
    rates = ds["ledger"].drop_duplicates("ticker").set_index("ticker")["r"]
    frozen = pd.DataFrame({"ticker": ["CCC"], "type": ["put"], "moneyness": [0.975], "h": [0.5]})
    pols = {"P0": {"kind": "none"}, "P1": ocfg["policies"]["P1"], "P2": {"kind": "rules"},
            "P3_frozen": {"kind": "frozen"}}
    hist = {"m": pd.Series([1.1, 1.2, 1.3, 1.25]), "sigma_hat": ds["tickers"].pivot_table(index="date", columns="ticker",
                                                                                          values="sigma_hat"),
            "prices": pd.DataFrame([ds["tickers"].set_index("ticker")["S_t"]] * 12)}
    kw = dict(rates=rates, m_bar=1.2, months=3, n=10, n_scenarios=400, seed=9, ocfg=ocfg, history=hist, frozen=frozen)
    a = paths.simulate_overlay_paths(model, sf, pols, **kw)
    b = paths.simulate_overlay_paths(model, sf, pols, **kw)
    pd.testing.assert_frame_equal(a, b)                             # deterministyczność przy ustalonym ziarnie
    assert a.notna().all().all() and a.shape == (400, len(sf) * len(pols))
    stats = paths.summary(a)
    p0 = stats.set_index(["subfund", "policy"])
    assert p0.loc[("SF1", "P1"), "es"] < p0.loc[("SF1", "P0"), "es"]  # put ogranicza ogon
    # P3_frozen tylko na CCC: SF1 bez zmian
    np.testing.assert_allclose(a["SF1|P3_frozen"], a["SF1|P0"])
