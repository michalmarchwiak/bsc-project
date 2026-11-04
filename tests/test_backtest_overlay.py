"""Nakładka opcyjna w głównym backteście: bliźniaki <strategia>_ov, realizacja z wypłatami opcji, konsolidacja."""

import numpy as np
import pandas as pd
import pytest

from pf.backtest import engine
from pf.backtest.realized import add_sid, realize_overlay
from pf.options import nav, portfolio
from pipelines import backtest as bt
from tests.test_backtest import smoke_configs, synthetic_panel

KAPPA = 0.03


def _legs(tickers, m=0.975, prem=0.012):
    return pd.DataFrame({"ticker": tickers, "type": "put", "moneyness": m, "h": 1.0, "premium_rel": prem,
                         "delta": -0.4})


def test_realize_overlay_matches_nav_formula():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2021-01-01", periods=70)
    loc = pd.DataFrame(rng.normal(0, 0.012, (70, 2)), index=idx, columns=["A.WA", "B"])
    fx = pd.Series(rng.normal(0, 0.006, 70), index=idx)
    pln = loc.assign(B=loc["B"] + fx)
    dates = pd.DatetimeIndex([idx[5], idx[27], idx[50]])
    w = pd.Series([0.5, 0.5], index=["A.WA", "B"])
    pos = pd.concat([_legs(["A.WA"]), _legs(["B"], m=1.0, prem=0.02).assign(type="call", delta=0.5)], ignore_index=True)
    D = float(w @ nav._per_ticker(pos, list(w.index), KAPPA)[0])
    sid = "m|ce_dep_ov|P3|base"
    W = pd.concat([pd.DataFrame({"date": d, "sid": sid, "ticker": w.index, "weight": (w / D).to_numpy()})
                   for d in dates[:2]])
    legs = pd.concat([pos.assign(date=d, sid=sid, subfund="SF1", W_ov=1.0, D=D, w=pos["ticker"].map(w))
                      for d in dates[:2]])
    daily, per = realize_overlay(pln, loc, W, legs, pd.Series({sid: 1}), dates, cost=0.001)
    assert len(per) == 2
    for j in range(2):
        sess = idx[(idx > dates[j]) & (idx <= dates[j + 1])]
        g = np.exp(loc.loc[sess].sum())
        G = np.exp(pln.loc[sess].sum())
        ref = nav.period_return(w, G, g, G / g, pos, KAPPA)["r_ov"]
        assert per["r_gross"].iloc[j] == pytest.approx(ref, rel=1e-10)
        dj = daily[daily["date"] == dates[j]]
        assert np.prod(1 + dj["r_gross"].astype(float)) - 1 == pytest.approx(ref, abs=1e-6)
    assert per["cost"].iloc[0] == pytest.approx(0.001 * float((w / D).sum()))
    assert per["overlay_cost"].iloc[0] == pytest.approx(1 - 1 / D)


def test_scenario_gross_and_stock_exposure():
    R = np.array([[0.10, 0.05, 0.02], [-0.20, -0.10, 0.01]])
    gross = portfolio.scenario_gross(R, ["A.WA", "B", "USDPLN"])
    np.testing.assert_allclose(gross["B"][0] * gross["B"][1], 1 + R[:, 1])
    np.testing.assert_allclose(gross["B"][1], 1 + R[:, 2])
    np.testing.assert_allclose(gross["A.WA"][1], 1.0)
    assert "USDPLN" not in gross
    from pf.opt.profiles import TwoLevelResult
    res = TwoLevelResult("ce", "P3", pd.Series({"SF1": 0.6, "SF1_OV": 0.4}),
                         {"SF1": pd.Series([0.5, 0.5], index=["A.WA", "B"])})
    pos = {"SF1": _legs(["A.WA", "B"])}
    x = portfolio.stock_exposure(res, pos, KAPPA)
    D = 1 + 0.012 * (1 + KAPPA)
    assert x.sum() + 0.4 * (1 - 1 / D) == pytest.approx(1.0)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ovrun")
    cfg, bcfg, pcfg, sf = smoke_configs()
    cfg = {**cfg, "dependence": {**cfg["dependence"], "models": ["gauss"]}}
    bcfg = {**bcfg, "grid": {**bcfg["grid"], "models": ["gauss"], "strategies": ["ce_dep", "maxret_es"]},
            "sensitivity": {**bcfg["sensitivity"], "enabled": False}}
    r = synthetic_panel(n=360, seed=3)
    rng = np.random.default_rng(4)
    fx = pd.Series(rng.normal(0, 0.006, len(r)), index=r.index, name="USDPLN")
    local = r.copy()
    for c in ("U1", "U2", "U3", "V1", "V2", "V3"):
        local[c] = r[c] - fx
    returns = r.assign(USDPLN=fx)
    dates = bt.all_dates(returns, cfg, bcfg, sf)[:4]
    positions = {dates[1]: {"SF2": _legs(["U1", "U2", "U3"], prem=0.004)},
                 dates[2]: {"SF2": _legs(["U1", "U2"], prem=0.004), "SF1": _legs(["P1.WA"], prem=0.004)}}
    ctx = {"positions": positions, "kappa": KAPPA, "strategies": ["ce_dep", "maxret_es"], "min_profile": "P4",
           "fx_name": "USDPLN", "subfunds": ["SF1", "SF2", "SF3"]}
    for i, t in enumerate(dates):
        nxt = dates[i + 1] if i + 1 < len(dates) else None
        engine.run_date(t, nxt, returns, cfg, bcfg, pcfg, sf, tmp, tmp / "cache", latest=nxt is None, overlay=ctx)
    out = bt.consolidate(tmp, returns, cfg, bcfg, pcfg, sf, rf=pd.Series(0.02, index=r.index), returns_local=local)
    return tmp, out, dates


def test_backtest_overlay_series_and_consolidation(run):
    tmp, out, dates = run
    per = out["periods"]
    ov = per[per["strategy"].str.endswith("_ov")]
    assert set(ov["profile"]) == {"P4", "P5"} and set(ov["strategy"]) == {"ce_dep_ov", "maxret_es_ov"}
    assert set(ov["variant"]) == {"base"} and ov["r_gross"].notna().all()
    base = per[per["strategy"].isin(["ce_dep", "maxret_es"]) & per["profile"].isin(["P4", "P5"])]
    # data bez pozycji opcyjnych: bliźniak ≡ szereg bazowy
    m = ov.assign(s=ov["strategy"].str[:-3]).merge(base, left_on=["date", "s", "profile", "model"],
                                                   right_on=["date", "strategy", "profile", "model"])
    first = m[m["date"] == dates[0]]
    np.testing.assert_allclose(first["r_gross_x"], first["r_gross_y"], atol=1e-12)
    legs = add_sid(pd.concat([pd.read_parquet(f) for f in sorted((tmp / "dates").glob("*/overlay.parquet"))]))
    W = out["weights_W"]
    ovW = W[W["subfund"].str.endswith("_OV")]
    assert set(ovW["subfund"]) <= {"SF1_OV", "SF2_OV"}
    if len(legs):
        assert (legs["D"].astype(float) > 1).all() and (legs["w"].astype(float) > 0).all()
        assert set(legs["subfund"]) <= {"SF1", "SF2"}
    # Σ x + B₀ = 1 dla każdego szeregu i daty
    wx = pd.read_parquet(tmp / "weights_x.parquet")
    for (d, sid), g in wx[wx["sid"].isin(set(ov["sid"]))].groupby(["date", "sid"]):
        lg = legs[(legs["date"] == d) & (legs["sid"] == sid)] if len(legs) else legs
        u = lg.drop_duplicates("subfund")
        B0 = float((u["W_ov"].astype(float) * (1 - 1 / u["D"].astype(float))).sum()) if len(lg) else 0.0
        assert g["weight"].sum() + B0 == pytest.approx(1.0, abs=1e-9)
    fc = out["forecasts"]
    assert fc.loc[fc["strategy"].str.endswith("_ov"), "es"].notna().all()
    assert not out["periods_sf"]["series"].str.contains("_ov").any()
    assert (tmp / "tests.parquet").exists()
