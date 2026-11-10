"""Uniwersum punktowo w czasie: konfiguracja, reguła wyboru składu, integracja z backtestem, benchmarki rynkowe."""

import numpy as np
import pandas as pd
import pytest

from pf import config, universe
from pf.backtest import engine
from pf.portfolio import load_subfunds
from pipelines import backtest as bt
from tests.test_backtest import smoke_configs, synthetic_panel

RCFG = {"window_days": 250, "min_obs": 200}


# --------------------------------------------------------------------------- konfiguracja

def test_universes_config_is_consistent(monkeypatch):
    import data_import

    cfg = config.load("subfunds")
    assert cfg["universe"] in cfg["universes"]
    for name, sfs in cfg["universes"].items():
        tickers = [t for spec in sfs.values() for t in spec["tickers"]]
        assert len(tickers) == len(set(tickers)), f"{name}: ticker w dwóch sub-funduszach"
        assert list(sfs) == [f"SF{i}" for i in range(1, 11)]
        for sf, spec in sfs.items():
            sel = spec.get("select")
            if sel:
                core = sel.get("core", [])
                assert set(core) <= set(spec["tickers"])
                assert 0 < sel["top"] <= len(spec["tickers"]) - len(core)
        quoted = [t for t in tickers if not t.endswith(".SP")]
        assert set(quoted) <= set(data_import.TICKERS)              # każdy ticker jest pobierany
    assert universe.uses_selection(cfg["universes"]["pit"]) and not universe.uses_selection(cfg["universes"]["fixed"])
    assert data_import.TICKERS[:len(data_import.TICKERS_BASE)] == data_import.TICKERS_BASE   # kolejność plików
    monkeypatch.setenv("PF_UNIVERSE", "fixed")
    assert load_subfunds() is cfg["universes"]["fixed"]
    monkeypatch.setenv("PF_UNIVERSE", "brak")
    with pytest.raises(KeyError):
        load_subfunds()


def test_market_benchmark_tickers_are_downloaded():
    import data_import

    series = config.load("backtest")["market_benchmarks"]["series"]
    assert {t for w in series.values() for t in w} <= set(data_import.TICKERS)
    assert data_import.STOOQ_FIRST <= set(data_import.TICKERS)


# --------------------------------------------------------------------------- reguła wyboru

def test_review_date():
    idx = pd.bdate_range("2019-06-03", "2022-03-31")
    assert universe.review_date(idx, pd.Timestamp("2021-01-04"), 1) == pd.Timestamp("2021-01-01")
    assert universe.review_date(idx, pd.Timestamp("2021-12-31"), 1) == pd.Timestamp("2021-01-01")
    assert universe.review_date(idx, pd.Timestamp("2022-01-03"), 1) == pd.Timestamp("2022-01-03")
    assert universe.review_date(idx, pd.Timestamp("2019-09-02"), 1) == idx[0]        # brak wcześniejszej rewizji
    assert universe.review_date(idx, pd.Timestamp("2021-08-02"), 7) == pd.Timestamp("2021-07-01")


def pit_setup(n=700, seed=5):
    """Pula 6 spółek w SF1 (wybór 2 + rdzeń X0), SF2 bez wyboru; obroty rosną z numerem spółki, w 2021 r. odwrotnie."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    cols = ["X0", "A1", "A2", "A3", "A4", "NEW", "B1", "B2", "USDPLN"]
    r = pd.DataFrame(rng.standard_normal((n, len(cols))) * 0.01, index=idx, columns=cols)
    r.loc[idx[:400], "NEW"] = np.nan                                 # debiut po 400 sesjach
    months = pd.period_range(idx[0], idx[-1], freq="M")
    level = {"X0": 1.0, "A1": 10.0, "A2": 20.0, "A3": 30.0, "A4": 40.0, "NEW": 1000.0, "B1": 5.0, "B2": 5.0}
    tv = pd.DataFrame({k: np.full(len(months), v) for k, v in level.items()}, index=months)
    tv.loc[tv.index >= "2021-01", ["A1", "A2", "A3", "A4"]] = [40.0, 30.0, 20.0, 10.0]
    sfs = {"SF1": {"name": "akcje", "select": {"top": 2, "core": ["X0"]},
                   "tickers": ["X0", "A1", "A2", "A3", "A4", "NEW"]},
           "SF2": {"name": "obligacje", "tickers": ["B1", "B2"]}}
    return r, tv, sfs


def test_selection_is_point_in_time():
    r, tv, sfs = pit_setup()
    uni = universe.Universe(sfs, r, RCFG, tv, {"lookback_months": 12, "review_month": 1})
    t20, t21, t21b = pd.Timestamp("2020-06-01"), pd.Timestamp("2021-01-01"), pd.Timestamp("2021-03-01")
    assert uni.review(t20) == pd.Timestamp("2020-01-01") and uni.review(pd.Timestamp("2019-09-02")) == pd.Timestamp(
        "2019-09-02")                                                # za mało danych przed rewizją 2019-01 → samo t
    assert uni.at(t20)["SF1"]["tickers"] == ["X0", "A3", "A4"]         # rdzeń + 2 największe obroty przed 2020-01
    assert uni.at(t20)["SF2"] is sfs["SF2"] and "select" not in uni.at(t20)["SF1"]
    # rewizja 2021-01: obroty 2021 r. (odwrócone) jeszcze nieznane; NEW ma za krótką historię (< min_obs)
    assert uni.at(t21)["SF1"]["tickers"] == ["X0", "A3", "A4"] and uni.at(t21b) == uni.at(t21)
    assert uni.at(pd.Timestamp("2021-10-01"))["SF1"]["tickers"] == ["X0", "A4", "NEW"]   # NEW dopuszczona w 2021 r.
    # zmiana obrotów po dacie rewizji nie zmienia składu
    tv2 = tv.copy()
    tv2.loc[tv2.index >= "2021-01"] = 0.0
    uni2 = universe.Universe(sfs, r, RCFG, tv2, {"lookback_months": 12, "review_month": 1})
    assert uni2.at(t21)["SF1"]["tickers"] == uni.at(t21)["SF1"]["tickers"]
    # zawężenie tabeli stóp: skład daty + aktywa spoza pul, kolejność kolumn pliku
    ret_t, sub_t = uni.view(t21)
    assert list(ret_t.columns) == ["X0", "A3", "A4", "B1", "B2", "USDPLN"] and sub_t is uni.at(t21)
    sched = uni.schedule(pd.DatetimeIndex([t20, t21, t21b]))
    assert sched["review"].nunique() == 2 and sched["date"].nunique() == 3 and set(sched["subfund"]) == {"SF1"}
    row = sched[(sched["ticker"] == "NEW") & (sched["date"] == t21)].iloc[0]
    assert not row["eligible"] and not row["selected"]


def test_selection_follows_turnover_and_eligibility():
    r, tv, sfs = pit_setup(n=900)
    uni = universe.Universe(sfs, r, RCFG, tv, {"lookback_months": 12, "review_month": 1})
    t22 = pd.Timestamp("2022-03-01")                                # rewizja 2022-01: obroty 2021 r., NEW dopuszczona
    assert uni.at(t22)["SF1"]["tickers"] == ["X0", "A1", "NEW"]
    # NEW uzyskuje min_obs w trakcie 2021 r.: wchodzi do składu bez czekania na rewizję (ranking z rewizji 2021-01)
    first = next(t for t in r.index[r.index >= "2021-01-01"] if "NEW" in uni.at(t)["SF1"]["tickers"])
    assert pd.Timestamp("2021-02-01") < first < pd.Timestamp("2021-12-31")
    assert uni.at(first)["SF1"]["tickers"] == ["X0", "A4", "NEW"] and uni.review(first) == pd.Timestamp("2021-01-01")
    assert universe.select_members(sfs["SF1"], {"X0", "A1"}, pd.Series({"A1": 1.0, "A2": 9.0})) == ["X0", "A1"]
    assert universe.select_members(sfs["SF1"], {"A1", "A2", "A3"}, pd.Series({"A1": 1.0, "A2": 0.0})) == ["A1"]


def test_universe_without_selection_is_identity():
    r, tv, sfs = pit_setup()
    plain = {"SF1": {"name": "a", "tickers": ["A1", "A2"]}, "SF2": sfs["SF2"]}
    uni = universe.Universe(plain, r, RCFG)
    ret_t, sub_t = uni.view(pd.Timestamp("2021-06-01"))
    assert ret_t is r and sub_t is plain and uni.schedule(r.index[-3:]).empty
    assert list(universe.restrict_to_pool(r, plain, extra=["USDPLN"]).columns) == ["A1", "A2", "B1", "B2", "USDPLN"]
    with pytest.raises(FileNotFoundError):
        universe.Universe(sfs, r, RCFG)                             # wybór składu bez tabeli obrotów


# --------------------------------------------------------------------------- integracja z backtestem

def test_backtest_uses_date_members_and_market_benchmarks(tmp_path):
    cfg, bcfg, pcfg, sf = smoke_configs()
    cfg = {**cfg, "dependence": {**cfg["dependence"], "models": ["gauss"]}}
    bcfg = {**bcfg, "grid": {**bcfg["grid"], "models": ["gauss"], "profiles": ["P3"], "strategies": ["ce"]},
            "forecast_model": "gauss", "sensitivity": {"enabled": False}, "options_overlay": {"enabled": False},
            "market_benchmarks": {"enabled": True, "series": {"idx_a": {"M1": 1.0}, "idx_mix": {"M1": 3.0, "M2": 1.0},
                                                             "idx_none": {"BRAK": 1.0}}}}
    r = synthetic_panel(n=420, seed=3)
    pool = {**sf, "SF2": {"name": "US", "select": {"top": 2}, "tickers": ["U1", "U2", "U3"]}}
    months = pd.period_range(r.index[0], r.index[-1], freq="M")
    tv = pd.DataFrame({"U1": 1.0, "U2": 3.0, "U3": 2.0}, index=months)
    uni = universe.Universe(pool, r, cfg["rebalance"], tv, {"lookback_months": 12, "review_month": 1})
    dates = bt.all_dates(r, cfg, bcfg, pool)[:3]
    for i, t in enumerate(dates):
        ret_t, sub_t = uni.view(t)
        s = engine.run_date(t, dates[i + 1] if i + 1 < len(dates) else None, ret_t, cfg, bcfg, pcfg, sub_t, tmp_path,
                            tmp_path / "cache", latest=i + 1 == len(dates))
        assert s["n_assets"] == 8                                   # U1 poza składem – nie wchodzi do modeli
    rng = np.random.default_rng(0)
    bench = pd.DataFrame(rng.standard_normal((len(r), 2)) * 0.01, index=r.index, columns=["M1", "M2"])
    out = bt.consolidate(tmp_path, r, cfg, bcfg, pcfg, pool, rf=pd.Series(0.02, index=r.index),
                         bench_returns=bench.assign(BRAK=np.nan))
    w = pd.read_parquet(tmp_path / "weights_tickers.parquet")
    assert "U1" not in set(w["ticker"]) and {"U2", "U3"} <= set(w["ticker"])
    per = out["periods"]
    mb = per[per["strategy"].str.startswith("idx_")]
    assert set(mb["sid"]) == {"-|idx_a|ALL|base", "-|idx_mix|ALL|base"}     # benchmark bez danych pominięty
    assert set(mb["profile"]) == {"ALL"} and mb["sample"].notna().all()
    one = mb[mb["sid"] == "-|idx_a|ALL|base"].sort_values("date")
    lag = int(bcfg["execution_lag"])
    i0, i1 = r.index.searchsorted(dates[0]) + lag, r.index.searchsorted(dates[1]) + lag
    assert one["r_gross"].iloc[0] == pytest.approx(np.expm1(bench["M1"].iloc[i0:i1].sum()))
    assert one["cost"].iloc[0] == pytest.approx(bcfg["transaction_cost"]) and one["cost"].iloc[1] == pytest.approx(0.0)
    assert out["tests"]["strategy"].str.startswith("idx_").sum() == 0       # bez prognoz – bez testów VaR/ES
    from pipelines import ranking as rk
    from pf.opt.profiles import load_profiles

    gam = {"P3": load_profiles(pcfg)["P3"].gamma}
    rcfg = {"mcs_size": 0.1, "mcs_reps": 100, "sharpe_test": {"block": 2, "reps": 49, "benchmark": "-|ew|ALL|base"},
            "composite_weights": {"ce_ann": 1.0}}
    tab = rk.rank_sample(per.assign(sample="calibration"), out["tests"], gam, rcfg, 0.975, None, "calibration")
    assert {"idx_a", "idx_mix", "ew", "ce"} <= set(tab["strategy"])


def test_run_id_depends_on_selection(monkeypatch):
    cfg, bcfg, pcfg, _ = smoke_configs()
    bcfg = {**bcfg, "options_overlay": {"enabled": False}}
    monkeypatch.setattr(bt, "run_id", lambda configs, path, length=12: bt.config_digest(configs)[:length])
    fixed, pit = load_subfunds("fixed"), load_subfunds("pit")
    a = bt.current_run_id(cfg, bcfg, pcfg, fixed)
    assert a == bt.current_run_id(cfg, {**bcfg, "market_benchmarks": {"enabled": False}}, pcfg, fixed)  # post-processing
    assert a != bt.current_run_id(cfg, bcfg, pcfg, pit)
