import numpy as np
import pandas as pd
import pytest

from pf.options import bs, iv

FILTERS = {"min_bid": 0.0, "max_rel_spread": 0.25, "require_open_interest": True, "max_stale_days": 3,
           "min_days_to_expiry": 7, "max_days_to_expiry": 400}


@pytest.mark.parametrize("is_call", [True, False])
def test_round_trip_price_iv_price(is_call):
    S = 100.0
    K = np.linspace(60, 150, 91)[None, :]
    T = np.array([7 / 365, 30 / 365, 0.5, 1.0, 2.0])[:, None]
    sigma = 0.1 + 0.5 * np.random.default_rng(1).random((5, 91))
    p = bs.price(S, K, T, 0.04, 0.012, sigma, is_call)
    lower, _ = iv.no_arbitrage_bounds(S, K, T, 0.04, 0.012, is_call)
    usable = (p - lower) > 1e-6 * S  # wartość czasowa rozróżnialna numerycznie
    s = iv.implied_vol(p, S, K, T, 0.04, 0.012, is_call)
    assert np.isfinite(s[usable]).all()
    p2 = bs.price(S, K, T, 0.04, 0.012, s, is_call)
    np.testing.assert_allclose(p2[usable], p[usable], atol=1e-6, rtol=0)
    np.testing.assert_allclose(s[usable & (p > 0.01)], sigma[usable & (p > 0.01)], atol=1e-6)


def test_prices_outside_arbitrage_bounds_give_nan():
    S, K, T, r, q = 100.0, 90.0, 0.5, 0.03, 0.0
    lower, upper = iv.no_arbitrage_bounds(S, K, T, r, q, True)
    out = iv.implied_vol(np.array([lower - 0.1, upper + 0.1, np.nan]), S, K, T, r, q, True)
    assert np.isnan(out).all()
    assert np.isnan(iv.implied_vol(1.0, S, K, 0.0, r, q, True))


def quotes():
    qd = pd.Timestamp("2024-03-01")
    return pd.DataFrame({
        "quote_date": qd,
        "expiration": pd.to_datetime(["2024-04-19"] * 6 + ["2024-03-04"]),
        "strike": [480, 480, 480, 480, 480, 480, 480],
        "call_put": ["P"] * 7,
        "bid": [5.0, 0.0, 5.0, 5.0, 5.0, 6.0, 5.0],
        "ask": [5.4, 0.2, 8.0, 5.4, 5.4, 5.0, 5.4],
        "openInterest": [10, 10, 10, 0, 10, 10, 10],
        "lastTradeDate": pd.to_datetime(["2024-03-01", "2024-03-01", "2024-03-01", "2024-03-01",
                                         "2024-02-20", "2024-03-01", "2024-03-01"], utc=True),
        "underlying_price": 510.0,
    })


def test_quote_filters_reasons():
    reasons = iv.quote_filters(quotes(), FILTERS).tolist()
    assert reasons == ["", "zero_bid", "wide_spread", "zero_oi", "stale_trade", "crossed", "short_expiry"]


def test_filters_skip_missing_optional_columns():
    q = quotes().drop(columns=["openInterest", "lastTradeDate"])
    assert iv.quote_filters(q, FILTERS).tolist()[3:5] == ["", ""]


def test_implied_vols_from_quotes_uses_mid():
    q = quotes().iloc[[0]].copy()
    T = (q["expiration"] - q["quote_date"]).dt.days.iloc[0] / 365
    target = 0.18
    mid = float(bs.price(510.0, 480.0, T, 0.05, 0.01, target, False))
    q["bid"], q["ask"] = mid - 0.05, mid + 0.05
    out = iv.implied_vols_from_quotes(q, FILTERS, 0.05, 0.01)
    assert out["reject"].iloc[0] == ""
    assert out["iv"].iloc[0] == pytest.approx(target, abs=1e-8)
