"""Agregacja panelu reszt (sumy k-dniowe) do estymacji kopuł oraz dane wejściowe wyboru składu."""

import numpy as np
import pandas as pd
import pytest

from pf import config
from pf.backtest import engine
from pf.models.dependence import fit_model
from pf.models.gof import pit_panel_aggregated
from pf.models.margins import aggregation_spec, residual_panel
from pf.models.rvine import pseudo_observations
from tests.test_backtest import synthetic_panel


def lagged_pair(n=6000, rho=0.8, seed=0):
    """x – rynek „wcześniejszy”, y – ten sam czynnik zarejestrowany z opóźnieniem jednej sesji (asynchroniczność)."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(n + 1)
    x = f[1:]
    y = rho * f[:-1] + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
    return pd.DataFrame({"X.WA": x, "Y": y}, index=pd.bdate_range("2000-01-03", periods=n))


def test_aggregation_spec_defaults():
    assert aggregation_spec(None) == (1, True) and aggregation_spec({}) == (1, True)
    assert aggregation_spec({"aggregation": {"days": 5, "overlap": False}}) == (5, False)
    assert aggregation_spec({"aggregation": {"days": 0}}) == (1, True)
    assert aggregation_spec(config.load("models")["dependence"])[0] >= 1


def test_residual_panel_aggregation_rules():
    idx = pd.bdate_range("2021-01-04", periods=12)
    z = pd.DataFrame({"A.WA": np.arange(12.0), "B": 10.0 * np.arange(12.0)}, index=idx)
    z.loc[idx[5], "B"] = np.nan                                       # święto USA: jedna sesja
    z.loc[idx[8:10], "A.WA"] = np.nan                                 # GPW zamknięta dwie sesje
    daily = residual_panel(z)
    assert len(daily) == 9 and residual_panel(z, agg_days=1).equals(daily)
    p2 = residual_panel(z, agg_days=2)
    assert p2.index[0] == idx[1] and idx[9] not in p2.index           # niepełna suma; okno bez sesji GPW
    assert idx[5] in p2.index and p2.loc[idx[5], "B"] == 40.0         # brak → 0, wiersz zostaje
    assert p2.loc[idx[4], "A.WA"] == 3.0 + 4.0 and p2.loc[idx[10], "A.WA"] == 10.0
    blocks = residual_panel(z, agg_days=3, overlap=False)
    assert list(blocks.index) == [idx[2], idx[5], idx[8], idx[11]]    # bloki rozłączne liczone od końca okna
    assert blocks.loc[idx[11], "B"] == 10.0 * (9 + 10 + 11)


def test_aggregated_copula_recovers_lagged_dependence():
    z = lagged_pair()
    cfg = config.load("models")
    rho = {}
    for k in (1, 5):
        panel = residual_panel(z, agg_days=k)
        model = fit_model("gauss", pseudo_observations(panel.to_numpy()), list(panel.columns), cfg)
        rho[k] = float(model.R[0, 1])
    assert abs(rho[1]) < 0.05                                         # zależność jednoczesna ≈ 0
    assert rho[5] == pytest.approx(0.8 * 4 / 5, abs=0.05)             # (k−1)/k · ρ dla opóźnienia 1
    # zależność sum 21-dniowych: dane ≈ 20/21·ρ; scenariusze iid z kopuły k = 5 są jej bliższe niż z kopuły dziennej
    target = np.corrcoef(z.rolling(21).sum().dropna().iloc[::21].to_numpy(), rowvar=False)[0, 1]
    assert abs(rho[5] - target) < 0.5 * abs(rho[1] - target)


def test_engine_reuses_margins_across_aggregations(tmp_path):
    cfg = config.load("models")
    cfg = {**cfg, "rebalance": {**cfg["rebalance"], "window_days": 250, "min_obs": 200}}
    r = synthetic_panel(n=300, seed=1)
    t = r.index[260]

    def with_agg(days):
        return {**cfg, "dependence": {**cfg["dependence"], "aggregation": {"days": days, "overlap": True}}}

    no_agg = {**cfg, "dependence": {k: v for k, v in cfg["dependence"].items() if k != "aggregation"}}
    d1 = engine.fit_margins_step(t, r, with_agg(1), cache_dir=tmp_path)
    assert not d1.cached["margins"] and (d1.folder / "z.parquet").exists() and (d1.folder / "panel.parquet").exists()
    d5 = engine.fit_margins_step(t, r, with_agg(5), cache_dir=tmp_path)
    assert d5.cached["margins"] and d5.folder == d1.folder and (d5.folder / "panel_agg5o.parquet").exists()
    assert len(d5.panel) == len(d1.panel) - 4
    np.testing.assert_allclose(d5.panel.to_numpy(), d1.panel.rolling(5).sum().dropna().to_numpy(), atol=1e-10)
    again = engine.fit_margins_step(t, r, with_agg(5), cache_dir=tmp_path)
    np.testing.assert_allclose(again.panel.to_numpy(), d5.panel.to_numpy())
    # klucze modeli: panel dzienny jak przed wprowadzeniem agregacji, wariant k = 5 – osobny
    assert engine.dependence_key(with_agg(1)) == engine.dependence_key(no_agg) != engine.dependence_key(with_agg(5))
    assert engine.panel_file(no_agg) == "panel.parquet"
    # pamięć podręczna sprzed zmiany (bez z.parquet): panel dzienny wczytywany, wariant k = 5 estymowany ponownie
    (d1.folder / "z.parquet").unlink()
    (d1.folder / "panel_agg5o.parquet").unlink()
    assert engine.fit_margins_step(t, r, no_agg, cache_dir=tmp_path).cached["margins"]
    assert not engine.fit_margins_step(t, r, with_agg(5), cache_dir=tmp_path).cached["margins"]


def test_oos_pseudo_observations_of_aggregates(tmp_path):
    cfg = config.load("models")
    cfg = {**cfg, "rebalance": {**cfg["rebalance"], "window_days": 250, "min_obs": 200},
           "dependence": {**cfg["dependence"], "aggregation": {"days": 5, "overlap": True}}}
    r = synthetic_panel(n=300, seed=2)
    t = r.index[260]
    dm = engine.fit_margins_step(t, r, cfg, cache_dir=None)
    oos = r.loc[r.index >= t].iloc[:21]
    U = pit_panel_aggregated(dm.margins, oos, dm.panel, 5, True)
    assert list(U.columns) == list(dm.panel.columns) and len(U) == 21 - 4
    assert ((U > 0) & (U < 1)).all().all() and 0.2 < float(U.to_numpy().mean()) < 0.8
    engine.fit_models_step(dm, cfg, ["gauss"])
    from pf.models.gof import oos_log_score

    assert np.isfinite(oos_log_score(dm.models["gauss"], U)).all()


# --------------------------------------------------------------------------- dane: obroty, tryb „tylko brakujące”

def ohlcv(idx, close, volume):
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Adj Close": close,
                         "Volume": volume}, index=pd.DatetimeIndex(idx, name="Date"))


def test_monthly_turnover_and_index_without_volume(tmp_path, monkeypatch):
    import data_manipulation as dm

    monkeypatch.setattr(dm, "RAW_DIR", tmp_path)
    (tmp_path / "ohlcv").mkdir()
    idx = pd.bdate_range("2020-01-01", "2020-02-28")
    vol = np.where(np.arange(len(idx)) % 10 == 0, 0, 100)             # co 10. sesja bez obrotu
    ohlcv(idx, 2.0, vol).to_csv(tmp_path / "ohlcv" / "AAA.csv")
    ohlcv(idx, 50.0, 0).to_csv(tmp_path / "ohlcv" / "IDX.WA.csv")     # indeks: wolumen zawsze 0
    tv = dm.monthly_turnover(["AAA", "IDX.WA", "BRAK"])
    assert list(tv.columns) == ["AAA"] and list(tv.index) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")]
    assert tv["AAA"].tolist() == [200.0, 200.0]
    assert len(dm.non_trading_days("AAA")) == int((vol == 0).sum())
    assert len(dm.non_trading_days("IDX.WA")) == 0                    # szereg bez wolumenu nie jest filtrowany


def test_missing_only_reuses_files_and_stooq_local_copy(tmp_path, monkeypatch):
    import data_import as di

    monkeypatch.setattr(di, "OUT_DIR", tmp_path)
    monkeypatch.setattr(di, "TICKERS", ["OLD", "NEW", "WIG20TR.WA"])
    (tmp_path / "ohlcv").mkdir()
    (tmp_path / "stooq").mkdir()
    idx = pd.bdate_range("2020-01-01", periods=5)
    ohlcv(idx, 10.0, 100).to_csv(tmp_path / "ohlcv" / "OLD.csv")
    pd.DataFrame([{"ticker": "OLD", "source": "yahoo"}]).to_csv(tmp_path / "coverage.csv", index=False)
    (tmp_path / "stooq" / "wig20tr.csv").write_text(
        "Data,Otwarcie,Najwyzszy,Najnizszy,Zamkniecie\n2020-01-02,1,1,1,3000\n2020-01-03,1,1,1,3030\n")
    calls = []

    def fake_yahoo(tickers, start, end):
        calls.append(list(tickers))
        return {t: ohlcv(idx, 20.0, 5) for t in tickers}

    monkeypatch.setattr(di, "fetch_yahoo", fake_yahoo)
    prices, sources = di.download_prices(missing_only=True)
    assert calls == [["NEW"]]                                         # OLD z dysku, indeks GPW ze Stooq (kopia lokalna)
    assert sources == {"OLD": "yahoo", "WIG20TR.WA": "stooq", "NEW": "yahoo"}
    assert prices["WIG20TR.WA"]["Adj Close"].tolist() == [3000.0, 3030.0]
    assert prices["OLD"]["Close"].iloc[0] == 10.0
    calls.clear()
    di.download_prices()
    assert calls == [["OLD", "NEW"]]                                  # pełne pobranie pomija tylko tickery Stooq
