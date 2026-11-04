"""Nakładka opcyjna – Etap 4: polityki P0–P3, LP vs brute force, determinizm, brak look-ahead."""

import itertools

import numpy as np
import pandas as pd
import pytest

from pf.backtest.data import rebalance_dates
from pf.options import calendar, engine, nav, policy
from pf.options.market import MarketData
from tests.options_synth import SUBFUNDS, models_config, overlay_config, synthetic_market

RCFG = {"q_cheap": 0.3, "q_expensive": 0.7, "v_max": 0.8, "put_cheap": 1.0, "put_trend": 0.975, "collar_put": 0.975,
        "collar_call": 1.05, "covered_call": 1.025}


@pytest.mark.parametrize("Q,T,V,kind", [
    (0.2, 0.05, 0.5, "put_cheap"), (0.5, -0.02, 0.5, "put"), (0.7, -0.02, 0.9, "put"), (0.8, -0.01, 0.5, "collar"),
    (0.8, 0.03, 0.5, "covered_call"), (0.8, 0.03, 0.9, "none"), (0.5, 0.03, 0.5, "none"),
    (np.nan, -0.1, 0.1, "none"), (0.75, np.nan, 0.1, "none")])
def test_rule_kind_table(Q, T, V, kind):
    assert policy.rule_kind(Q, T, V, RCFG) == kind


def test_expanding_percentile_min_history_and_no_lookahead():
    x = pd.Series([3.0, 1.0, 2.0, 5.0, 4.0, 0.5])
    p = policy.expanding_percentile(x, min_history=2)
    assert np.isnan(p.iloc[:2]).all()
    assert p.iloc[2] == pytest.approx(2 / 3) and p.iloc[3] == 1.0 and p.iloc[5] == pytest.approx(1 / 6)
    y = x.copy()
    y.iloc[4:] = [100.0, -100.0]
    np.testing.assert_array_equal(policy.expanding_percentile(y, 2).iloc[:4], p.iloc[:4])


def _legs(tickers, puts=(0.975,), calls=(1.025,), prem=0.015):
    rows = []
    for t in tickers:
        rows += [{"ticker": t, "type": "put", "moneyness": m, "premium_rel": prem, "delta": -0.4, "in_band": True}
                 for m in puts]
        rows += [{"ticker": t, "type": "call", "moneyness": m, "premium_rel": prem * 0.8, "delta": 0.4,
                  "in_band": True} for m in calls]
    return pd.DataFrame(rows)


def _scen(S=3000, seed=0, sig=(0.07, 0.05)):
    rng = np.random.default_rng(seed)
    z = rng.standard_t(4, size=(S, 2)) / np.sqrt(2)
    g = np.exp(0.002 + np.array(sig) * (0.6 * z[:, [0]] + 0.8 * z))
    f = np.column_stack([np.ones(S), np.exp(rng.normal(0, 0.02, S))])
    return g, f, g * f


@pytest.mark.parametrize("lam", [0.5, 2.0, 5.0])
def test_lp_vs_brute_force_two_tickers(lam):
    tickers = ["A.WA", "B"]
    g, f, G = _scen()
    w = np.array([0.5, 0.5])
    legs = _legs(tickers)
    R0, A = policy.lp_matrix(w, G, g, f, tickers, legs, 0.03)
    h, info = policy.solve_lp(R0, A, [], lam, 0.975)
    assert info["status"] == "ok"
    # wartość funkcji celu z LP = bezpośrednia ocena E[R] − λ ES w rozwiązaniu
    assert info["objective"] == pytest.approx(policy.objective(R0 + A @ h, lam, 0.975), abs=1e-10)
    grid = np.linspace(0, 1, 6)
    best = max(policy.objective(R0 + A @ np.array(x), lam, 0.975) for x in itertools.product(grid, repeat=4))
    assert info["objective"] >= best - 1e-10
    # optimum lokalne: żadne przesunięcie o ±0,05 w otoczeniu rozwiązania nie poprawia funkcji celu
    near = max(policy.objective(R0 + A @ np.clip(h + d, 0, 1), lam, 0.975)
               for d in itertools.product([-0.05, 0, 0.05], repeat=4))
    assert info["objective"] >= near - 1e-10


def test_lp_group_constraints_and_inaction_band():
    tickers = ["A.WA", "B"]
    g, f, G = _scen(sig=(0.12, 0.10))
    w = np.array([0.6, 0.4])
    legs = _legs(tickers, puts=(0.95, 0.975, 1.0), calls=(1.0, 1.025, 1.05), prem=0.002)   # tania ochrona
    pos, info = policy.lp_positions(w, G, g, f, tickers, legs, 0.03, 3.0, 0.975, 0.0)
    assert info["active"] and info["gain"] > 0
    sums = pos.groupby(["ticker", "type"])["h"].sum()
    assert (sums <= 1 + 1e-9).all()
    pos2, info2 = policy.lp_positions(w, G, g, f, tickers, legs, 0.03, 3.0, 0.975, epsilon=1.0)
    assert pos2.empty and not info2["active"] and info2["gain"] == pytest.approx(info["gain"])
    # bardzo droga ochrona i bardzo tanie calle: brak nakładki
    exp = _legs(tickers, puts=(0.975,), calls=(1.025,), prem=0.2)
    exp.loc[exp["type"] == "call", "premium_rel"] = 1e-6
    pos3, _ = policy.lp_positions(w, G, g, f, tickers, exp, 0.03, 1.0, 0.975, 0.0)
    assert pos3.empty


def test_static_and_rule_positions():
    legs = _legs(["A.WA", "B"], puts=(0.975, 1.0), calls=(1.025, 1.05))
    p1 = policy.static_positions(legs, ["A.WA", "B"], {"put": 0.975, "h_put": 0.5})
    assert list(p1["h"]) == [0.5, 0.5] and set(p1["type"]) == {"put"}
    sig = pd.DataFrame({"ticker": ["A.WA", "B"], "Q": [0.8, 0.8], "T": [-0.02, 0.03], "V": [0.5, 0.5]})
    p2 = policy.rule_positions(legs, sig, RCFG)
    assert nav.overlay_kind(p2[p2["ticker"] == "A.WA"]) == "collar"
    assert nav.overlay_kind(p2[p2["ticker"] == "B"]) == "covered_call"
    assert p2.loc[p2["ticker"] == "B", "moneyness"].iloc[0] == 1.025


# --------------------------------------------------------------------------- krok daty na danych syntetycznych

@pytest.fixture(scope="module")
def setup():
    market = synthetic_market()
    cfg, ocfg = models_config(), overlay_config()
    dates = rebalance_dates(market.index, cfg["rebalance"]["window_days"])
    periods = calendar.overlay_periods(market.index, dates, lag=1)
    return market, cfg, ocfg, periods


def _truncate(m: MarketData, d) -> MarketData:
    cut = lambda x: x.loc[x.index <= d]  # noqa: E731
    return MarketData(cut(m.r_local), cut(m.r_pln), cut(m.fx), cut(m.vol), cut(m.rate_pln), cut(m.rate_usd),
                      cut(m.spx))


def test_date_step_deterministic_and_no_lookahead(setup):
    market, cfg, ocfg, periods = setup
    per = periods.iloc[2].to_dict()
    a = engine.date_step(per, market, cfg, ocfg, SUBFUNDS, seed=11)
    b = engine.date_step(per, market, cfg, ocfg, SUBFUNDS, seed=11)
    pd.testing.assert_frame_equal(a["decisions"], b["decisions"])
    pd.testing.assert_frame_equal(a["ledger"], b["ledger"])
    assert set(a["ledger"]["subfund"]) == {"SF1", "SF2", "SF8"}                 # SF10 wyłączony
    assert a["ledger"]["S_next"].notna().all()
    # zmiana danych po t_j (obcięcie i inne dane po t_open) nie zmienia wycen ani decyzji w t_j
    trunc = engine.date_step(per, _truncate(market, per["open"]), cfg, ocfg, SUBFUNDS, seed=11)
    cols = ["K", "S_t", "F", "r", "sigma_imp", "premium", "delta", "gamma", "vega", "fx_t", "in_band"]
    pd.testing.assert_frame_equal(a["ledger"][cols].reset_index(drop=True), trunc["ledger"][cols].reset_index(drop=True))
    pd.testing.assert_frame_equal(a["decisions"], trunc["decisions"])
    assert trunc["ledger"]["S_next"].isna().all()
    lp = a["lp_info"]
    assert set(lp["policy"]) == set(engine.lp_policies(ocfg)) >= {"P3", "P3_lam1", "P3_lam3"}
    assert {"base", "sigma+3pp", "kappa=0.05"} <= set(lp["variant"])
    assert np.isfinite(a["state"]["m"].iloc[0])


def test_consolidation_p012_and_nav(setup):
    market, cfg, ocfg, periods = setup
    parts = [engine.date_step(periods.iloc[j].to_dict(), market, cfg, ocfg, SUBFUNDS, seed=11)
             for j in range(6)]
    L = pd.concat([p["ledger"] for p in parts], ignore_index=True)
    state = pd.concat([p["state"] for p in parts], ignore_index=True)
    tick = pd.concat([p["tickers"] for p in parts], ignore_index=True)
    sig = engine.signals_table(state, tick, market, ocfg)
    assert sig.loc[sig["date"] == state["date"].iloc[0], "Q"].isna().all()     # za krótka historia
    dec = pd.concat([engine.p012_decisions(L, sig, ocfg)] + [p["decisions"] for p in parts], ignore_index=True)
    per, dout = engine.nav_table(L, dec, ocfg)
    base = per[per["variant"] == "base"]
    p0 = base[base["policy"] == "P0"]
    np.testing.assert_allclose(p0["r_ov"], p0["r_base"], atol=1e-15)
    assert (p0["kind"] == "none").all()
    p1 = base[base["policy"] == "P1"]
    assert (p1["kind"] == "put").all() and (p1["premium_paid"] > 0).all()
    # bazowa stopa SF = średnia brutto-stóp tickerów w PLN (wagi równe)
    d0 = L["date"].iloc[0]
    one = L[(L["date"] == d0) & (L["subfund"] == "SF2")].drop_duplicates("ticker")
    exp = float(np.mean(one["S_next"] / one["S_t"] * one["fx_next"] / one["fx_t"]) - 1)
    got = p0[(p0["date"] == d0) & (p0["subfund"] == "SF2")]["r_base"].iloc[0]
    assert got == pytest.approx(exp, rel=1e-12)
    # wariant κ zmienia tylko koszt (te same h P1)
    k5 = per[(per["policy"] == "P1") & (per["variant"] == "kappa=0.05")]
    assert (k5["kappa_cost"].to_numpy() > p1["kappa_cost"].to_numpy()).all()
    assert {"w", "e", "kind"} <= set(dout.columns) and (dout["e"] >= -1e-12).all()   # K^P ≤ K^C ⇒ e ≥ 0


def test_robust_margins_refit_low_volatility():
    from pf.models.margins import fit_margins
    from pf.options import scenarios

    rng = np.random.default_rng(8)
    idx = pd.bdate_range("2015-01-01", periods=900)
    vol = 0.0008 * np.exp(0.3 * np.sin(np.arange(900) / 50))
    w = pd.DataFrame({"LOW": vol * rng.standard_t(6, 900) + 0.00005, "HIGH": 0.015 * rng.standard_t(6, 900)},
                     index=idx)
    fits, z = fit_margins(w, scale=100.0, jobs=1)
    fits, z2, refit = scenarios.robust_margins(w, fits, z, 100.0)
    assert refit == ["LOW"]
    f = fits["LOW"]
    assert f.scale == pytest.approx(1 / w["LOW"].std()) and scenarios.plausible(f, 1.0)
    assert z2["LOW"].std() == pytest.approx(1.0, abs=0.1)
    from pf.options.implied_vol import garch_vol
    assert garch_vol(f, 21) == pytest.approx(w["LOW"].std() * np.sqrt(252), rel=0.5)
    bad = fits["HIGH"].__class__(**{**fits["HIGH"].to_dict(), "mu": -3.5e6, "alpha": 1.0})
    assert not scenarios.plausible(bad, 1.5)
