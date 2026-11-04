"""Nakładka opcyjna – Etap 1: stopa USD, kalendarz okresów opcji, dane rynkowe w datach t_j."""

import numpy as np
import pandas as pd
import pytest

from pf.data import usd_rate
from pf.options import calendar, market
from tests.options_synth import USD_TICKERS, synthetic_market


def test_parse_fred_and_combine():
    text = "observation_date,DTB3\n2020-01-02,1.52\n2020-01-03,.\n2020-01-06,1.50\n2020-01-07,1.49\n"
    fred = usd_rate.parse_fred_csv(text)
    assert list(fred.index.strftime("%F")) == ["2020-01-02", "2020-01-06", "2020-01-07"]
    irx = pd.Series([1.51, 1.48], index=pd.to_datetime(["2020-01-06", "2020-01-08"]))
    out = usd_rate.combine(irx, fred)
    assert out.loc["2020-01-06", "source"] == "irx" and out.loc["2020-01-06", "rate"] == pytest.approx(0.0151)
    assert out.loc["2020-01-02", "source"] == "fred_dtb3" and out.index.is_monotonic_increasing
    assert not out.index.duplicated().any()


def test_continuous_rate_discount_factor():
    y, d, tau = 0.05, 30, 21 / 252
    r = usd_rate.continuous_rate(y, d, tau, 365)
    assert np.exp(-r * tau) == pytest.approx(1 / (1 + y * d / 365), rel=1e-14)


def test_overlay_periods_lag1_and_lag0():
    idx = pd.bdate_range("2021-01-01", "2021-04-30")
    dates = pd.DatetimeIndex(["2021-02-01", "2021-03-01", "2021-04-01"])
    p1 = calendar.overlay_periods(idx, dates, lag=1, horizon_days=21)
    first = p1.iloc[0]
    assert first["open"] == pd.Timestamp("2021-02-01") and first["close"] == pd.Timestamp("2021-03-01")
    assert first["n_sessions"] == len(idx[(idx > "2021-02-01") & (idx <= "2021-03-01")])
    assert first["tau"] == pytest.approx(first["n_sessions"] / 252)
    assert first["window_end"] == pd.Timestamp("2021-01-29")
    last = p1.iloc[-1]
    assert pd.isna(last["close"]) and last["n_sessions"] == 21
    p0 = calendar.overlay_periods(idx, dates, lag=0)
    assert p0.iloc[0]["open"] == pd.Timestamp("2021-01-29") and p0.iloc[0]["close"] == pd.Timestamp("2021-02-26")
    assert len(calendar.sessions_of(idx, p1.iloc[0], 1)) == first["n_sessions"]


def test_market_levels_fx_and_gross_identity():
    m = synthetic_market(n=200)
    assert market.currency("PKO.WA") == "PLN" and market.currency("OTS.SP") == "PLN" and market.currency("AAPL") == "USD"
    sessions = m.index[50:71]
    d0, d1 = m.index[49], m.index[70]
    tickers = USD_TICKERS[:2] + ["AAA.WA"]
    g = m.gross(sessions, tickers, pln=False)
    G = m.gross(sessions, tickers, pln=True)
    s0, s1 = m.spot(d0, tickers), m.spot(d1, tickers)
    np.testing.assert_allclose(s1 / s0, g, rtol=1e-12)
    f = m.fx_of(d1, tickers) / m.fx_of(d0, tickers)
    np.testing.assert_allclose(G, g * f, rtol=1e-10)           # G = g·f
    assert f["AAA.WA"] == 1.0
    # dzień bez notowania: cena i kurs tickera z ostatniej sesji
    hol = m.r_local.index[m.r_local["AAA.WA"].isna()]
    hol = hol[hol > m.index[5]][0]
    prev = m.r_local.index[m.r_local.index < hol][-1]
    assert m.spot(hol, ["AAA.WA"]).iloc[0] == m.spot(prev, ["AAA.WA"]).iloc[0]


def test_fx_log_returns_on_calendar():
    m = synthetic_market(n=120)
    r = m.fx_log_returns()
    np.testing.assert_allclose(np.exp(r.iloc[1:].sum()), m.fx.iloc[-1] / m.fx.iloc[0], rtol=1e-12)


def test_asof_staleness():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-01", "2020-01-10"]))
    assert market.asof(s, "2020-01-09") == 1.0
    assert np.isnan(market.asof(s, "2020-01-09", max_stale_days=3))
    assert np.isnan(market.asof(s, "2019-12-31"))
