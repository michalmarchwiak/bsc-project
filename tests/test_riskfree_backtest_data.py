"""Etap 0: stopa wolna od ryzyka, opóźnienie wykonania, run_id, kontrola kompletności danych (bez sieci)."""

import numpy as np
import pandas as pd
import pytest

from pf.backtest import data as bd
from pf.data import quality, riskfree

STOOQ_PL = """Data,Otwarcie,Najwyzszy,Najnizszy,Zamkniecie
2024-01-02,5.86,5.86,5.86,5.86
2024-01-03,5.85,5.85,5.85,5.85
2024-01-03,5.85,5.85,5.85,5.84
2024-01-05,5.83,5.83,5.83,5.83
"""
STOOQ_EN = """Date,Open,High,Low,Close
2024-01-02,5.86,5.86,5.86,5.86
"""
NBP_ARCHIVE = """<?xml version="1.0" encoding="UTF-8"?>
<stopy_procentowe_archiwum>
  <pozycje obowiazuje_od="2023-09-07">
    <pozycja id="ref" nazwa="Stopa referencyjna" oprocentowanie="6,00"/>
    <pozycja id="lom" nazwa="Stopa lombardowa" oprocentowanie="6,50"/>
  </pozycje>
  <pozycje obowiazuje_od="2023-10-05">
    <pozycja id="ref" nazwa="Stopa referencyjna" oprocentowanie="5,75"/>
  </pozycje>
</stopy_procentowe_archiwum>
"""
NBP_FLAT = """<stopy_procentowe><tabela>
<pozycja id="ref" oprocentowanie="5,75" obowiazuje_od="2023-10-05"/>
</tabela></stopy_procentowe>"""


def test_period_rate_act365():
    rate = pd.Series([0.0586, 0.0584], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    t0, t1 = pd.Timestamp("2024-01-03"), pd.Timestamp("2024-02-01")
    assert riskfree.period_rate(rate, t0, t1) == pytest.approx(0.0584 * 29 / 365)
    # stopa z ostatniego fixingu ≤ t0 (sobota → fixing piątkowy)
    assert riskfree.rate_at(rate, pd.Timestamp("2024-01-06")) == 0.0584
    assert riskfree.period_rate(0.05, t0, t0 + pd.Timedelta(days=365)) == pytest.approx(0.05)
    with pytest.raises(KeyError):
        riskfree.rate_at(rate, pd.Timestamp("2023-12-29"))


def test_parse_stooq_and_nbp():
    s = riskfree.parse_stooq_csv(STOOQ_PL, "plopln1m")
    assert list(s.index.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03", "2024-01-05"]
    assert s.loc["2024-01-03"] == 5.84  # duplikat – ostatnia obserwacja
    assert riskfree.parse_stooq_csv(STOOQ_EN).iloc[0] == 5.86
    with pytest.raises(ValueError):
        riskfree.parse_stooq_csv("Przekroczony dzienny limit wywolan", "plopln1m")
    nbp = riskfree.parse_nbp_rates_xml(NBP_ARCHIVE)
    assert nbp.to_dict() == {pd.Timestamp("2023-09-07"): 6.0, pd.Timestamp("2023-10-05"): 5.75}
    assert riskfree.parse_nbp_rates_xml(NBP_FLAT).iloc[0] == 5.75


def test_combine_fills_tail_with_nbp_and_has_no_gaps():
    wibor = pd.Series([5.80, 5.70], index=pd.to_datetime(["2023-10-02", "2023-10-06"]))
    nbp = riskfree.parse_nbp_rates_xml(NBP_ARCHIVE)
    table, rep = riskfree.combine(wibor, nbp, "2023-10-02", "2023-10-31", max_ffill_days=5)
    assert table["rate"].notna().all() and rep["n_missing"] == 0
    assert table.loc["2023-10-03", "rate"] == pytest.approx(0.058)        # ffill w luce ≤ 5 dni
    assert table.loc["2023-10-11", "rate"] == pytest.approx(0.057)        # 5 dni po fixingu
    assert table.loc["2023-10-12", "source"] == "nbp_ref"                 # WIBOR „skończył się”
    assert table.loc["2023-10-12", "rate"] == pytest.approx(0.0575)
    assert rep["wibor_last"] == "2023-10-06" and rep["n_filled_nbp"] == 14
    assert rep["filled_ranges"] == [["2023-10-12", "2023-10-31"]]
    # bez stopy NBP – dni po końcu WIBOR zostają puste i są raportowane
    _, rep2 = riskfree.combine(wibor, None, "2023-10-02", "2023-10-31", max_ffill_days=5)
    assert rep2["n_missing"] == 14


def test_holding_sessions_execution_lag():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    t, t_next = pd.Timestamp("2020-02-03"), pd.Timestamp("2020-03-02")
    lag0 = bd.holding_sessions(idx, t, t_next, 0)
    lag1 = bd.holding_sessions(idx, t, t_next, 1)
    assert lag0[0] == t and lag0[-1] == pd.Timestamp("2020-02-28")         # [t, t_next)
    assert lag1[0] == pd.Timestamp("2020-02-04") and lag1[-1] == t_next   # (t, t_next]
    assert len(lag0) == len(lag1)
    assert bd.holding_bounds(idx, t, t_next, 1) == (t, t_next)
    assert bd.holding_bounds(idx, t, t_next, 0) == (pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28"))
    assert bd.holding_sessions(idx, t, None, 1)[-1] == idx[-1]
    r = pd.DataFrame({"X": np.log1p(np.where(idx == t, 0.10, 0.0))}, index=idx)
    # stopa z sesji t należy do okresu tylko przy L = 0
    assert bd.realized_returns(r, t, t_next, lag=0)["X"] == pytest.approx(0.10)
    assert bd.realized_returns(r, t, t_next, lag=1)["X"] == pytest.approx(0.0)


def test_sample_label_and_ranges():
    bcfg = {"calibration": ["2008-01-01", "2020-12-31"], "holdout": ["2021-01-01", "2026-10-29"]}
    assert bd.sample_label(pd.Timestamp("2020-12-01"), bcfg) == "calibration"
    assert bd.sample_label(pd.Timestamp("2021-01-04"), bcfg) == "holdout"
    assert bd.sample_label(pd.Timestamp("2007-12-03"), bcfg) is None
    dates = pd.DatetimeIndex(["2007-12-03", "2008-01-02", "2021-01-04"])
    assert list(bd.in_range(dates, bcfg["calibration"])) == [pd.Timestamp("2008-01-02")]


def test_run_id_depends_on_config_and_data(tmp_path):
    f = tmp_path / "r.csv"
    f.write_text("Date,X\n2020-01-01,0.1\n")
    a = bd.run_id({"x": 1, "y": [1, 2]}, f)
    assert len(a) == 12 and a == bd.run_id({"y": [1, 2], "x": 1}, f)    # kolejność kluczy bez znaczenia
    assert a != bd.run_id({"x": 2, "y": [1, 2]}, f)
    f.write_text("Date,X\n2020-01-01,0.2\n")
    assert a != bd.run_id({"x": 1, "y": [1, 2]}, f)


def test_quality_reports():
    cov = pd.DataFrame({"ticker": ["A", "B", "C", "D"], "source": ["yahoo", "stooq", "MISSING", "yahoo"],
                        "first_date": ["2005-01-03", "2012-04-11", None, "2005-01-03"],
                        "last_date": ["2026-10-29", "2026-10-28", None, "2020-12-31"], "n_obs": [5000, 3000, 0, 4000]})
    rep = quality.coverage_report(cov, "2005-01-01", "2026-10-29")
    assert not rep["ok"] and rep["missing"] == ["C"] and rep["stale"] == ["D"] and "B" in rep["late_start"]
    idx = pd.bdate_range("2005-01-03", "2026-10-29")
    ret = pd.DataFrame({"A": 0.0, "B": np.nan}, index=idx)
    r = quality.returns_report(ret, ["A", "B", "E"], "2005-01-01", "2026-10-29")
    assert not r["ok"] and r["empty_columns"] == ["B"] and r["absent"] == ["E"] and r["covers_range"]
    assert quality.returns_report(ret[["A"]], ["A"], "2005-01-01", "2026-10-29")["ok"]


def test_stooq_price_parser_rejects_messages():
    import data_import

    assert data_import.parse_stooq_prices("Przekroczony dzienny limit wywolan") is None
    df = data_import.parse_stooq_prices(STOOQ_PL.replace("Zamkniecie", "Zamkniecie,Wolumen")
                                        .replace("5.86\n", "5.86,100\n"))
    assert df is not None and list(df.columns) == data_import.OHLCV_COLUMNS
    assert data_import.stooq_symbol("AVB") == "avb.us"


@pytest.mark.network
def test_wibor_symbol_available():
    from pf import config

    cfg = config.load("backtest")["riskfree"]
    s, _ = riskfree.fetch_wibor({**cfg, "start": "2024-01-01", "end": "2024-03-01"})
    assert 0.0 < s.median() < 20.0


@pytest.mark.network
def test_nbp_reference_available():
    from pf import config

    s, _ = riskfree.fetch_nbp_reference(config.load("backtest")["riskfree"])
    assert s.index.min() < pd.Timestamp("2005-01-01") and 0.0 <= s.iloc[-1] < 25.0
