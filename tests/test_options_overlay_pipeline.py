"""Nakładka opcyjna – Etap 6: walidacja BXM/PPUT, miary backtestu, pipeline end-to-end na danych syntetycznych."""

import json

import numpy as np
import pandas as pd
import pytest

from pf import config
from pf.backtest import overlay as bto
from pf.options import validation
from pipelines import options_monthly as om
from tests.options_synth import SUBFUNDS, models_config, overlay_config, synthetic_market

ICFG = {"vix_tenors_days": {"VIX9D": 9, "VIX": 30}, "atm_adj": 1.0, "sigma_min": 0.01, "sigma_max": 1.5, "beta": 0.0}


def _spx(n=800, seed=4):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    spx = pd.Series(2000 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))), index=idx)
    tr = spx * np.exp(0.015 / 252 * np.arange(n))
    vol = pd.DataFrame({"VIX": 18.0, "VIX9D": 16.0}, index=idx)
    y = pd.Series(0.02, index=idx)
    return spx, tr, vol, y


def test_strike_rules():
    assert validation.strike(2003.0, 1.0, 5.0, "ceil") == 2005.0
    assert validation.strike(2005.0, 1.0, 5.0, "ceil") == 2005.0
    assert validation.strike(2003.0, 0.95, 5.0, "floor") == 1900.0


@pytest.mark.parametrize("name", ["BXM", "PPUT"])
def test_replication_tracks_itself_and_signs(name):
    spx, tr, vol, y = _spx()
    spec = config.load("options")["monthly_overlay"]["validation"][name]
    rep = validation.replicate_monthly(spx, tr, vol, y, spec, ICFG)
    assert len(rep) >= 30 and (rep["premium"] > 0).all()
    assert rep["d0"].dt.dayofweek.isin([4, 0]).mean() > 0.9         # rolowanie w 3. piątek (lub kolejna sesja)
    if name == "BXM":
        assert (rep["K"] >= rep["S0"]).all()
    else:
        assert (rep["K"] <= 0.95 * rep["S0"] + 1e-9).all()
    lv = pd.Series(np.concatenate([[100.0], 100 * np.cumprod(1 + rep["r_replica"])]),
                   index=pd.DatetimeIndex([rep["d0"].iloc[0]] + list(rep["d1"])))
    tab = validation.tracking(rep, lv, {"x": ["2015-03-01", "2016-01-01"]})
    allp = tab[tab["period"] == "all"].iloc[0]
    assert allp["te_ann_pct"] == pytest.approx(0.0, abs=1e-10) and allp["corr"] == pytest.approx(1.0)
    assert "stress_x" in set(tab["period"])


def test_sf_metrics_and_stress():
    d = pd.date_range("2008-01-01", periods=30, freq="MS")
    rng = np.random.default_rng(0)
    rb = rng.normal(0.01, 0.05, 30)
    per = pd.DataFrame({"date": d, "subfund": "SF1", "policy": "P1", "variant": "base", "r_base": rb,
                        "r_ov": rb * 0.8, "rf": 0.003, "premium_paid": 0.01, "premium_received": 0.0,
                        "kappa_cost": 0.0003, "kind": ["put"] * 20 + ["none"] * 10, "sample": "calibration"})
    m = bto.sf_metrics(per)
    row = m[m["sample"] == "all"].iloc[0]
    assert row["share_overlay"] == pytest.approx(20 / 30) and row["premium_net"] == pytest.approx(0.30)
    st = bto.stress_table(per, {"2008": ["2008-09-01", "2009-03-31"]})
    assert st.iloc[0]["n_months"] == 7
    assert st.iloc[0]["ret_ov"] == pytest.approx(np.prod(1 + per["r_ov"].iloc[8:15]) - 1)


def test_pipeline_end_to_end(tmp_path):
    market = synthetic_market()
    cfg, ocfg = models_config(), overlay_config()
    bcfg, pcfg = config.load("backtest"), config.load("profiles")
    periods = om.overlay_periods(market, cfg, bcfg, SUBFUNDS).iloc[:6]
    ctx = {"market": market, "cfg": cfg, "ocfg": ocfg, "subfunds": SUBFUNDS}
    om.run_ledger(periods, tmp_path, 1, False, ctx)
    done = tmp_path / "dates" / periods["date"].iloc[0].strftime("%Y-%m-%d") / "done.json"
    first = json.loads(done.read_text())
    mtime = done.stat().st_mtime
    om.run_ledger(periods, tmp_path, 1, False, ctx)                 # wznowienie: data nie jest liczona ponownie
    assert done.stat().st_mtime == mtime and json.loads(done.read_text()) == first
    cons = om.consolidate(periods, market, cfg, bcfg, ocfg, tmp_path)
    for f in ("ledger.parquet", "overlay_decisions.parquet", "subfunds_overlay_nav.parquet", "signals.parquet"):
        assert (tmp_path / f).exists()
    led = pd.read_parquet(tmp_path / "ledger.parquet")
    assert set(led["ticker"].astype(str)) <= {t for sf in ("SF1", "SF2", "SF8") for t in SUBFUNDS[sf]["tickers"]}
    assert len(led) == len(led.drop_duplicates(["date", "ticker", "type", "moneyness"]))
    nav_p = cons["nav"]
    assert set(nav_p["policy"]) == set(ocfg["policies"]) and nav_p["rf"].notna().all()
    pp = om.run_portfolio(periods, market, cfg, bcfg, ocfg, pcfg, SUBFUNDS, tmp_path, 1, False, ctx)
    assert set(pp["variant"]) == {"base", "ov"} and len(pp) == periods["t_next"].notna().sum() * 3 * 2 * 2
    annual = om.run_annual(cons, market, cfg, ocfg, SUBFUNDS, tmp_path)
    assert set(annual["policy"]) == set(ocfg["annual_mc"]["policies"])
    assert om.run_validation(ocfg, tmp_path, root=tmp_path) is None
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["reports/options_validation"]["params"]["status"] == "skipped"
    assert {"options/ledger", "subfunds_overlay_nav", "options/portfolio_periods", "options/annual_mc"} <= set(manifest)
    assert manifest["options/ledger"]["params"]["band"] == ocfg["band"]
    path = om.write_report(cons, pp, None, annual, ocfg, pcfg, tmp_path)
    md = path.read_text()
    assert "hipotetyczne" in md and "## Portfele" in md and "Scenariusze roczne" in md and "| SF1 | P3 |" in md
    again = om.load_consolidated(tmp_path)
    assert len(again["ledger"]) == len(led) and len(again["decisions"]) == len(cons["decisions_out"])
