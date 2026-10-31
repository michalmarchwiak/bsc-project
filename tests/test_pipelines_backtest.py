"""Integracja Etapów 4–6 na syntetycznym przebiegu: konsolidacja → MC ex post → ranking, raporty, wykresy."""

import numpy as np
import pandas as pd
import pytest

from pf.backtest import engine
from pipelines import backtest as bt
from pipelines import expost_mc as em
from pipelines import ranking as rk
from tests.test_backtest import smoke_configs, synthetic_panel


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run")
    cfg, bcfg, pcfg, sf = smoke_configs()
    r = synthetic_panel(n=560, seed=21)
    dates = bt.all_dates(r, cfg, bcfg, sf)[:13]
    bcfg = {**bcfg, "calibration": [str(dates[0].date()), str(dates[7].date())],
            "holdout": [str(dates[8].date()), str(dates[-1].date())],
            "sensitivity": {**bcfg["sensitivity"], "kappa_grid": {"values": [0.0, 1.0], "strategies": ["ce_dep"]}}}
    for i, t in enumerate(dates):
        nxt = dates[i + 1] if i + 1 < len(dates) else None
        engine.run_date(t, nxt, r, cfg, bcfg, pcfg, sf, tmp, tmp / "cache", latest=nxt is None)
    out = bt.consolidate(tmp, r, cfg, bcfg, pcfg, sf, rf=pd.Series(0.02, index=r.index))
    return tmp, cfg, bcfg, pcfg, sf, out


def test_expost_mc_pipeline(run):
    tmp, cfg, bcfg, pcfg, sf, out = run
    ecfg = {"horizon": 21, "batch": 2000, "n_min": 2000, "n_max": 4000, "rel_se": 0.05, "main_model": "rvine",
            "main_strategy": "ce_dep"}
    uni, multi = em.build_tasks(tmp, bcfg, ecfg, 0.975, cfg)
    assert len(uni) > 0 and len(multi) > 0
    rows = []
    for task in [u for u in uni if u[2] == "all"][:6]:
        rows += em.univariate_task(task)
    rows += em.multivariate_task(multi[0])
    es = pd.DataFrame(rows)
    assert {"fhs", "stationary_bootstrap", "multivariate_W_last", "multivariate_W_mean"} <= set(es["method"])
    assert (es["es"] > es["var"] - 1e-12).all()
    tab = em.comparison(es, tmp, bcfg)
    assert tab["es_hist"].notna().any() and "es_exante" in tab
    md = em.report(tab, ecfg)
    assert "MC ex post" in md


def test_ranking_pipeline(run, tmp_path, monkeypatch):
    tmp, cfg, bcfg, pcfg, sf, out = run
    from pf.opt.profiles import load_profiles

    rcfg = {"mcs_size": 0.1, "mcs_reps": 200, "sharpe_test": {"block": 2, "reps": 99, "benchmark": "-|ew|ALL|base"},
            "composite_weights": {"ce_ann": 3, "sharpe": 1, "max_dd": 1, "turnover": 0.5}, "test_level": 0.05,
            "main_series": {"model": "rvine", "strategy": "ce_dep"}}
    profiles = load_profiles(pcfg)
    gam = {k: v.gamma for k, v in profiles.items()}
    tabs = {s: rk.rank_sample(out["periods"], out["tests"], gam, rcfg, 0.975, None, s) for s in rk.SAMPLES}
    assert len(tabs["holdout"]) == 5 * (2 * 4 + 1 + 1)
    for s in rk.SAMPLES:
        assert "## P3" in rk.ranking_markdown(tabs[s], s, profiles, rcfg, 4)
    dep, text = rk.dependence_ranking(out["gof"], out["logscore"], out["tests"], tabs, rcfg, bcfg)
    assert set(dep["model"]) >= {"rvine", "gauss"} and "Ranking modeli" in text
    timings = pd.read_parquet(tmp / "timings.parquet")
    md = rk.summary_markdown(tmp, out["periods"], out["tests"], out["flags"], out["scenario_stats"], timings, tabs,
                             profiles, gam, bcfg)
    assert "Wrażliwość" in md and "κ wybrane" in md
    monkeypatch.setattr(rk, "FIG_DIR", tmp_path)
    monkeypatch.setattr(rk, "ROOT", tmp_path)
    files = rk.figures(out["periods"], out["weights_W"], out["forecasts"], profiles, rcfg, bcfg)
    assert len(files) == 4 and all((tmp_path / f).exists() for f in files)
